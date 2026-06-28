"""SolaraViz dashboard components for the fishing ABM."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import solara
from matplotlib.figure import Figure

from mesa.visualization import Slider, SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import SpaceRendererComponent
from mesa.visualization.utils import update_counter

PANEL_STYLE = {"width": "100%", "min-width": "0", "overflow": "hidden"}
ROW_STYLE = {"width": "100%", "gap": "12px", "align-items": "stretch"}

BASE_MODEL_PARAMS = {
    "rng": {"type": "SliderInt", "value": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    "width": {
        "type": "SliderInt",
        "value": 100,
        "min": 20,
        "max": 200,
        "step": 10,
        "label": "Grid width",
    },
    "height": {
        "type": "SliderInt",
        "value": 100,
        "min": 20,
        "max": 200,
        "step": 10,
        "label": "Grid height",
    },
    "n_agents": {
        "type": "SliderInt",
        "value": 80,
        "min": 10,
        "max": 300,
        "step": 5,
        "label": "Boats",
    },
    "v": Slider("Comm range v", value=3, min=1, max=8, step=1),
    "r": Slider("Growth rate r", value=0.1, min=0.01, max=0.5, step=0.005),
    "q": Slider("Catchability q", value=0.3, min=0.05, max=1.0, step=0.01),
    "c": Slider("Travel cost c", value=0.05, min=0.0, max=0.3, step=0.005),
    "beta": Slider("Logit rationality β", value=2.0, min=0.1, max=10.0, step=0.1),
    "C_birth": Slider("Reproduction threshold", value=2.0, min=0.5, max=5.0, step=0.1),
    "sigma": Slider("Mutation σ (on λ)", value=0.5, min=0.0, max=1.0, step=0.05),
    "patch_scale": Slider("Patch scale (frac. of grid)", value=0.1, min=0.03, max=0.3, step=0.01),
    "finders_share": Slider("Finder's share f", value=0.5, min=0.1, max=0.9, step=0.05),
}


def _build_renderer(model):
    renderer = (
        SpaceRenderer(model, backend="matplotlib")
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
    )
    renderer.post_process = post_process_space
    renderer.render()
    return renderer


def _show_figure(fig):
    fig.set_constrained_layout(True)
    solara.FigureMatplotlib(
        fig, format="png", bbox_inches="tight", dependencies=[update_counter.value]
    )
    plt.close(fig)


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        x=agent.pos[0],
        y=agent.pos[1],
        color="tab:blue" if not agent.scrounge else "tab:red",
        marker="o",
        size=35,
        zorder=2,
    )


def propertylayer_portrayal(layer):
    if layer.name == "fish density":
        vmax = max(float(layer.data.max()), float(layer.data.min()) + 1e-6, 1.0)
        return PropertyLayerStyle(
            colormap="viridis",
            vmin=0,
            vmax=vmax,
            alpha=0.75,
            colorbar=True,
        )
    return PropertyLayerStyle(color="gray", alpha=0.3, colorbar=False)


def post_process_space(ax):
    ax.set_title("Fish density (background) and producer/scrounger roles (dots)", fontsize=10)
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markersize=7,
            label="Producer",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="tab:red",
            markersize=7,
            label="Scrounger",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)


@solara.component
def WealthHistogram(model, figsize=(4.5, 2.4)):
    update_counter.get()
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    caps = [a.capital for a in model.agents]
    if caps:
        ax.hist(caps, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(
        model.C_birth, color="green", ls="--", lw=1, label="reproduce threshold"
    )
    ax.set_xlabel("Capital C")
    ax.set_ylabel("Boats")
    ax.set_title("Wealth distribution", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    _show_figure(fig)


@solara.component
def TimeSeriesPlot(model, measure, ylabel=None, figsize=(4.5, 2.2)):
    update_counter.get()
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if isinstance(measure, str):
        ax.plot(df.loc[:, measure], linewidth=1.5)
        ax.set_ylabel(ylabel or measure, fontsize=9)
        ax.set_title(ylabel or measure, fontsize=10)
    else:
        for name, color in measure.items():
            ax.plot(df.loc[:, name], label=name, color=color, linewidth=1.5)
        ax.legend(loc="best", fontsize=7)
        ax.set_ylabel(ylabel or "Value", fontsize=9)
        ax.set_title(ylabel or "Time series", fontsize=10)
    ax.set_xlabel("Step", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(labelsize=8)
    _show_figure(fig)


@solara.component
def ParameterGuide(parameter_guide: str):
    update_counter.get()
    with solara.Details("Parameter guide & legend"):
        solara.Markdown(parameter_guide)


@solara.component
def LiveMetricsPanel(model):
    """Wealth histogram and fleet time series for the live view."""
    update_counter.get()
    with solara.Column(style={"width": "100%", "gap": "10px", **PANEL_STYLE}):
        WealthHistogram(model)
        TimeSeriesPlot(model, "Bankruptcies")
        TimeSeriesPlot(model, "Boats")


def make_diagnostics_panel(parameter_guide: str):
    @solara.component
    def DiagnosticsPanel(model):
        update_counter.get()
        with solara.Column(style={"width": "100%", "gap": "12px"}):
            with solara.Row(style=ROW_STYLE):
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(model, "Mean fish density")
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(model, "Mean lambda", ylabel="Mean loss aversion λ")
            with solara.Row(style=ROW_STYLE):
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(model, "Std lambda", ylabel="Std loss aversion λ")
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(model, "Scrounge rate")
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(
                        model,
                        {
                            "Network mean degree": "tab:blue",
                            "Network LCC fraction": "tab:green",
                            "Network components": "tab:orange",
                        },
                        ylabel="Info network (PS links)",
                    )
            with solara.Row(style=ROW_STYLE):
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    TimeSeriesPlot(model, "Wealth Gini")
                with solara.Column(style={"flex": "1", **PANEL_STYLE}):
                    ParameterGuide(parameter_guide)

    return DiagnosticsPanel


def make_fishing_dashboard(parameter_guide: str):
    DiagnosticsPanel = make_diagnostics_panel(parameter_guide)

    @solara.component
    def FishingDashboard(model):
        update_counter.get()
        tab_index, set_tab_index = solara.use_state(0)
        renderer = solara.use_memo(lambda: _build_renderer(model), [id(model)])

        with solara.v.Tabs(v_model=tab_index, on_v_model=set_tab_index):
            solara.v.Tab(children=["Live"])
            solara.v.Tab(children=["Diagnostics"])

        with solara.v.Window(v_model=tab_index):
            with solara.v.WindowItem():
                if tab_index == 0:
                    with solara.Row(style={**ROW_STYLE, "align-items": "flex-start"}):
                        with solara.Column(style={"flex": "7", **PANEL_STYLE}):
                            SpaceRendererComponent(model, renderer)
                        with solara.Column(style={"flex": "5", **PANEL_STYLE}):
                            LiveMetricsPanel(model)
            with solara.v.WindowItem():
                if tab_index == 1:
                    DiagnosticsPanel(model)

    return FishingDashboard


def create_dashboard(model_cls, model_params, name: str, parameter_guide: str):
    """Build a SolaraViz page for a fishing model class."""
    model = model_cls()
    FishingDashboard = make_fishing_dashboard(parameter_guide)
    return SolaraViz(
        model,
        components=[(FishingDashboard, 0)],
        model_params=model_params,
        name=name,
        play_interval=150,
    )
