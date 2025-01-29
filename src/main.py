import typer
from trogon import Trogon
from typer.main import get_group

import src.commands.app_analyzer as analyzer

app = typer.Typer(
    name="dataset_app",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)
app.add_typer(analyzer.app, name="analyzer")


@app.command()
def tui(ctx: typer.Context):
    Trogon(get_group(app), click_context=ctx).run()


@app.callback()
def explain():
    """

    Analyzer App: A CLI tool for image dataset statistical analysis.

    * check: to check the provided configuration file.

    * launch: to launch the GUI for the analysis using the provided configuration file.

    ---

    Build using Trogon and Typer for the CLI and script parts,
    OpenCV and NumPy for the low-level image processing.
    """


if __name__ == "__main__":
    app()
