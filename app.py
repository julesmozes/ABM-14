from fishing_viz import BASE_MODEL_PARAMS, create_dashboard
from model import FishingModel

PARAMETER_GUIDE = """
**Visual legend**
- **Background heatmap:** fish biomass density per cell (0 to K=1)
- **Boat dots:** blue = producer (searches privately, broadcasts finds), red = scrounge (joins neighbours' intel)
- **Wealth histogram:** green dashed line = reproduction threshold C_birth

**Parameters**
- **Grid width / height:** lattice dimensions (cells); reset to apply
- **Boats (N):** initial fleet size
- **Comm range v:** Chebyshev radius for crowding, information exchange, and the PS network
- **Finder's share f:** producer's fraction of pooled catch `q·D` when scroungers join the same patch; scroungers split `(1−f)`
- **Growth rate r:** intrinsic logistic regrowth rate of fish
- **Catchability q:** fraction of fish harvested when a boat fishes a cell
- **Travel cost c:** capital spent per grid cell moved
- **Logit rationality β:** inverse noise in destination and role choice (higher = more greedy)
- **Reproduction threshold C_birth:** capital required to spawn one offspring
- **Mutation σ:** std dev of Gaussian mutation on loss aversion λ at reproduction
- **Patch scale:** spatial width of Gaussian fish hotspots as a fraction of the grid

**Fixed constants** (see MODEL.md): perception radius v_see=1, ρ=0.88, K=1, m=0.02, 6 resource patches. **Loss aversion λ** is per-agent, evolves via birth–death selection + σ (bounds [1, 5]); roles are plastic each step.
"""

page = create_dashboard(
    FishingModel,
    BASE_MODEL_PARAMS,
    name="Fishing Model",
    parameter_guide=PARAMETER_GUIDE,
)
page  # noqa
