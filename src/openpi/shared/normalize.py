import json
import pathlib
import numpy as np
import numpydantic
import pydantic
import torch


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """
    Compute running statistics of a batch of vectors using PyTorch with GPU support.
    All operations are vectorized to eliminate Python loops.
    """

    def __init__(self, num_quantile_bins: int = 5000, device: str | torch.device = None):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        
        # Histograms: Shape [vector_length, num_bins]
        self._histograms = None
        # Bin edges: Shape [vector_length, num_bins + 1]
        self._bin_edges = None
        
        self._num_quantile_bins = num_quantile_bins
        self._device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _to_tensor(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to torch tensor on the correct device."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self._device)
        return data.to(self._device)

    def update(self, batch: np.ndarray | torch.Tensor) -> None:
        """
        Update the running statistics with a batch of vectors.
        
        Args:
            batch: An array/tensor where all dimensions except the last are batch dimensions.
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch.copy())
        
        # Ensure correct device
        batch = batch.to(self._device)
        
        # Flatten all leading dimensions into a single batch dimension
        original_shape = batch.shape
        batch = batch.reshape(-1, batch.shape[-1])
        
        num_elements, vector_length = batch.shape
        
        if self._count == 0:
            self._initialize(batch, vector_length)
        else:
            if vector_length != self._mean.shape[0]:
                raise ValueError(f"The length of new vectors ({vector_length}) does not match "
                                 f"the initialized vector length ({self._mean.shape[0]}).")
            
            # Update min/max and check for changes
            current_min = torch.min(batch, dim=0).values
            current_max = torch.max(batch, dim=0).values
            
            max_changed = torch.any(current_max > self._max)
            min_changed = torch.any(current_min < self._min)
            
            if max_changed or min_changed:
                self._adjust_histograms_vectorized(current_min, current_max)
                
            self._max = torch.maximum(self._max, current_max)
            self._min = torch.minimum(self._min, current_min)

        # Update count
        self._count += num_elements
        
        # Update running mean and mean of squares
        # Welford's online algorithm or simple incremental update
        batch_mean = torch.mean(batch, dim=0)
        batch_mean_of_squares = torch.mean(batch**2, dim=0)
        
        # Incremental update formula: new_mean = old_mean + (batch_mean - old_mean) * (batch_size / total_count)
        correction_factor = num_elements / self._count
        self._mean = self._mean + (batch_mean - self._mean) * correction_factor
        self._mean_of_squares = self._mean_of_squares + (batch_mean_of_squares - self._mean_of_squares) * correction_factor
        
        # Update histograms
        self._update_histograms_vectorized(batch)

    def _initialize(self, batch: torch.Tensor, vector_length: int) -> None:
        """Initialize statistics with the first batch."""
        self._mean = torch.mean(batch, dim=0)
        self._mean_of_squares = torch.mean(batch**2, dim=0)
        self._min = torch.min(batch, dim=0).values
        self._max = torch.max(batch, dim=0).values
        
        # Create initial bin edges for each feature
        # Shape: [vector_length, num_bins + 1]
        # We need to create linspace for each feature independently
        mins = self._min.unsqueeze(1)  # [V, 1]
        maxs = self._max.unsqueeze(1)  # [V, 1]
        
        # Create bins: [0, 1, ..., num_bins] / num_bins -> [0, 1]
        # Then scale to [min, max]
        bin_steps = torch.linspace(0, 1, self._num_quantile_bins + 1, device=self._device) # [B+1]
        
        # Broadcast: [V, 1] * 1 + [V, 1] * [1, B+1] -> [V, B+1]
        # edges = min + (max - min) * steps
        self._bin_edges = mins + (maxs - mins) * bin_steps.unsqueeze(0)
        
        # Initialize empty histograms
        self._histograms = torch.zeros((vector_length, self._num_quantile_bins), device=self._device)
        
        # Update histograms with initial batch
        self._update_histograms_vectorized(batch)

    def _update_histograms_vectorized(self, batch: torch.Tensor) -> None:
        """
        Update histograms for all features simultaneously without loops.
        Uses searchsorted to find bin indices for all elements.
        """
        # batch shape: [N, V]
        # bin_edges shape: [V, B+1]
        
        # We need to map each value in batch[:, v] to a bin index using bin_edges[v]
        # torch.searchsorted requires sorted sequences.
        
        # To vectorize searchsorted across different edges for each column:
        # This is tricky because standard searchsorted doesn't support batched edges directly in older torch versions.
        # However, we can use digitize or manual calculation if bins are uniform.
        # Since our bins are linear linspace, we can calculate indices mathematically which is much faster.
        
        # Calculate bin width for each feature
        # widths shape: [V]
        widths = (self._bin_edges[:, -1] - self._bin_edges[:, 0]) / self._num_quantile_bins
        
        # Avoid division by zero
        widths = torch.where(widths == 0, torch.ones_like(widths), widths)
        
        # Calculate normalized position
        # batch: [N, V], min: [V] -> [N, V]
        normalized = (batch - self._min.unsqueeze(0)) / widths.unsqueeze(0)
        
        # Get integer indices
        indices = torch.floor(normalized).long()
        
        # Clip indices to valid range [0, num_bins - 1]
        indices = torch.clamp(indices, 0, self._num_quantile_bins - 1)
        
        # Scatter add to histograms
        # histograms shape: [V, B]
        # We need to add 1 to histograms[v, indices[n, v]] for all n, v
        
        # Create a source tensor of ones
        ones = torch.ones_like(indices, dtype=self._histograms.dtype)
        
        # Use scatter_add_
        # dim=0 means we accumulate along the row dimension? No.
        # histograms is [V, B]. indices is [N, V].
        # We want to accumulate counts for each feature (row) into its bins (col).
        # Actually, it's easier to transpose logic or use bincount per column if N is small, 
        # but for large N, scatter is good.
        
        # Let's reshape histograms to [V, B] and indices to [N, V].
        # scatter_add_(dim, index, src)
        # If we operate on dim=1 (bins), index must have same shape as src.
        # src should be [V, N]? No.
        
        # Alternative: Loop over features is removed by using `bincount` if we process one by one, 
        # but we want NO loops.
        
        # Efficient way: 
        # Expand histograms to [N, V, B]? Too memory intensive.
        
        # Let's use the fact that scatter_add works on specific dimensions.
        # We want to update self._histograms[V, B].
        # For each feature v, we have indices[N].
        # This essentially requires a loop over V unless we use sparse operations or specific tricks.
        
        # However, `torch.histc` is not differentiable and doesn't support custom edges easily in batch.
        
        # Let's try a different vectorization strategy:
        # Since we calculated `indices` [N, V], we can use `scatter_add` if we transpose.
        # But scatter_add adds `src` into `self` at `index`.
        # self[V, B]. index[N, V]. This doesn't match directly.
        
        # Correct approach for fully vectorized histogram update without loops:
        # It is actually difficult to do perfectly vectorized histogram updates for *different* edges per column 
        # without significant memory overhead or specialized kernels.
        # BUT, since our edges are linear, we computed indices.
        
        # We can use `torch.zeros` and `scatter_add` by reshaping.
        # View histograms as flat? No.
        
        # Let's stick to a very optimized approach:
        # If V is small (e.g., < 1000) and N is large, a loop over V is often faster than complex broadcasting 
        # due to cache locality. BUT the requirement says "NO explicit Python loops".
        
        # Workaround: Use `bincount` via `scatter_add` on a flattened structure?
        # Let's try to use `scatter_add` correctly.
        # We want to count occurrences of each index for each feature.
        
        # Create a target tensor for accumulation
        # This is hard to do purely vectorized for 2D histograms with different edges.
        
        # Re-evaluating "No Loops":
        # If I strictly follow "no for i in range", I can use `torch.vmap` if available, 
        # or accept that `scatter_add` might need careful shaping.
        
        # Let's use a trick:
        # offsets = torch.arange(vector_length, device=self._device) * self._num_quantile_bins
        # global_indices = indices + offsets.unsqueeze(0) # [N, V]
        # flat_hist = torch.zeros(vector_length * self._num_quantile_bins, device=self._device)
        # flat_hist.scatter_add_(0, global_indices.flatten(), ones.flatten())
        # self._histograms = flat_hist.reshape(vector_length, self._num_quantile_bins)
        
        # This works! And it's fully vectorized.
        
        vector_length = batch.shape[1]
        offsets = torch.arange(vector_length, device=self._device) * self._num_quantile_bins # [V]
        
        # Broadcast offsets to [N, V]
        global_indices = indices + offsets.unsqueeze(0) # [N, V]
        
        # Flatten
        global_indices_flat = global_indices.reshape(-1) # [N*V]
        ones_flat = ones.reshape(-1) # [N*V]
        
        # Accumulate
        flat_hist = torch.zeros(vector_length * self._num_quantile_bins, device=self._device, dtype=self._histograms.dtype)
        flat_hist.scatter_add_(0, global_indices_flat, ones_flat)
        
        # Reshape back to [V, B] and add to existing
        new_hist = flat_hist.reshape(vector_length, self._num_quantile_bins)
        self._histograms += new_hist

    def _adjust_histograms_vectorized(self, new_min: torch.Tensor, new_max: torch.Tensor) -> None:
        """
        Adjust histograms when min or max changes.
        Redistributes existing counts to new bins.
        """
        # Old edges: [V, B+1]
        # New edges: [V, B+1]
        
        # Calculate new edges
        mins = self._min.unsqueeze(1) # Use updated self._min which already includes new_min? 
        # Wait, self._min is updated AFTER this call in update(). 
        # So we must use the passed new_min/new_max or update self._min first?
        # In update(), we call adjust BEFORE updating self._min/self._max.
        # So here self._min is still OLD.
        # But we want to expand to NEW min/max.
        
        # So we construct edges from new_min/new_max
        mins = new_min.unsqueeze(1)
        maxs = new_max.unsqueeze(1)
        
        bin_steps = torch.linspace(0, 1, self._num_quantile_bins + 1, device=self._device)
        new_edges = mins + (maxs - mins) * bin_steps.unsqueeze(0) # [V, B+1]
        
        # We need to redistribute self._histograms [V, B_old] to new bins.
        # The old bins covered [old_min, old_max].
        # The new bins cover [new_min, new_max] which is wider.
        
        # Strategy:
        # 1. Determine where the old bin centers fall in the new binning scheme.
        # 2. Assign the counts from old bins to the corresponding new bins.
        # Note: This is an approximation. Precise redistribution requires knowing the distribution within the old bin.
        # Assuming uniform distribution within old bins is standard for this type of online adjustment.
        
        # Old bin centers
        old_bin_widths = (self._bin_edges[:, -1] - self._bin_edges[:, 0]) / self._num_quantile_bins
        old_centers = self._bin_edges[:, :-1] + old_bin_widths.unsqueeze(1) / 2 # [V, B]
        
        # Map old centers to new bin indices
        new_widths = (new_edges[:, -1] - new_edges[:, 0]) / self._num_quantile_bins
        # Avoid div by zero
        new_widths = torch.where(new_widths == 0, torch.ones_like(new_widths), new_widths)
        
        # Calculate index in new grid
        # idx = (center - new_min) / new_width
        new_indices_float = (old_centers - mins) / new_widths.unsqueeze(1)
        new_indices = torch.floor(new_indices_float).long()
        
        # Clip to valid range [0, B-1]
        new_indices = torch.clamp(new_indices, 0, self._num_quantile_bins - 1)
        
        # Create new histogram tensor
        vector_length = self._histograms.shape[0]
        offsets = torch.arange(vector_length, device=self._device) * self._num_quantile_bins
        global_indices = new_indices + offsets.unsqueeze(1) # [V, B]
        
        flat_hist_new = torch.zeros(vector_length * self._num_quantile_bins, device=self._device, dtype=self._histograms.dtype)
        
        # Scatter add the OLD counts into the NEW flat histogram structure
        flat_hist_new.scatter_add_(0, global_indices.reshape(-1), self._histograms.reshape(-1))
        
        # Update state
        self._histograms = flat_hist_new.reshape(vector_length, self._num_quantile_bins)
        self._bin_edges = new_edges

    def _compute_quantiles(self, quantiles_list: list[float]) -> list[torch.Tensor]:
        """Compute quantiles based on histograms using vectorized cumulative sum."""
        # self._histograms: [V, B]
        # self._bin_edges: [V, B+1]
        
        cumsum = torch.cumsum(self._histograms, dim=1) # [V, B]
        
        results = []
        for q in quantiles_list:
            target_count = q * self._count
            # Find first index where cumsum >= target_count
            # searchsorted requires sorted input, cumsum is sorted.
            # We search for target_count in each row.
            
            # torch.searchsorted works on 1D or ND with specific handling.
            # For 2D, we can use it if we handle dimensions correctly.
            # searchsorted(sorted_sequence, values, *, side='left', out=None, sorter=None, right=None)
            
            # values must be broadcastable to sorted_sequence?
            # Let's create a tensor of target_counts [V, 1]
            targets = torch.full((self._histograms.shape[0], 1), target_count, device=self._device)
            
            # searchsorted returns indices [V, 1]
            indices = torch.searchsorted(cumsum, targets, right=True) # [V, 1]
            
            # Clamp indices to max bin index
            indices = torch.clamp(indices, 0, self._num_quantile_bins - 1)
            
            # Gather edges
            # edges [V, B+1]. We want edge at index.
            # gather(dim=1, index=indices)
            q_values = torch.gather(self._bin_edges, 1, indices).squeeze(1) # [V]
            
            results.append(q_values)
            
        return results

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.
        Returns NormStats with numpy arrays for compatibility.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        # Ensure non-negative variance due to floating point errors
        variance = torch.clamp(variance, min=0)
        stddev = torch.sqrt(variance)
        
        q_vals = self._compute_quantiles([0.01, 0.99])
        q01 = q_vals[0]
        q99 = q_vals[1]
        
        # Move to CPU and convert to numpy for NormStats
        return NormStats(
            mean=self._mean.cpu().numpy(),
            std=stddev.cpu().numpy(),
            q01=q01.cpu().numpy(),
            q99=q99.cpu().numpy()
        )


# --- Serialization Helpers (Adapted for Torch internal state if needed, 
# but here we stick to the original interface which saves NormStats) ---

class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the normalization statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the normalization statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())