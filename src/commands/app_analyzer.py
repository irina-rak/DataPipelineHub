from os import listdir, makedirs
from os.path import isdir, isfile, join
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from pydantic import ValidationError, BaseModel, ConfigDict, Field

from src.console import console
from src.dataset_analysis.utils import (
    get_histograms,
    get_images,
    get_images_stats,
    save_stats,
)
from src.utils.path import get_configurations


class ConfigDataset(BaseModel):
    """A Pydantic model for the pre-processing configuration file.

    Attributes:
    -----------
        dirs (dict): Dictionary containing the paths to the input,
            output and log directories.
        parameters (dict): Dictionary containing the parameters of the
            preprocessing steps.
        labels (dict): Dictionary containing the labels of the
            different classes.
    """

    root_dir: Path = Field(..., description="The root directory of the dataset.")
    dirs: dict = Field()
    # preprocessing: dict = Field()

    model_config = ConfigDict(extra="forbid")


def check_and_build_server_config(conf_loaded: dict):
    _conf = ConfigDataset(**conf_loaded)
    console.log(_conf)
    conf = dict(_conf)
    root_dir = conf["root_dir"]
    conf_dirs = conf["dirs"]
    # conf_preprocessing = conf["preprocessing"]
    return conf, root_dir, conf_dirs


app = typer.Typer(pretty_exceptions_show_locals=False, rich_markup_mode="rich")


@app.callback()
def preprocessing():
    """The pre-processing command line interface.

    It is made of two commands:

    * The command launch launches the pre-processing pipeline using the provided configuration file.
    * The command check checks if the provided configuration file satisfies the Pydantic constraints.

    """


@app.command(name="check")
def check_server_config(config: Annotated[Path, typer.Argument()]) -> None:
    """Check the provided server configuration file.

    The command loads the configuration file and checks the validity of the configuration using Pydantic.
    If the configuration is alright with respect to ConfigServer Pydantic BaseModel, nothing happens.
    Otherwise, raises the ValidationError by Pydantic -- which is quite verbose and should be useful understanding the issue with the configuration provided.

    You may pass optional parameters (in addition to the configuration file itself) to override the parameters given in the configuration.

    Parameters
    ----------
        config (Path): the Path to the configuration file.

    Raises
    ------
        ValidationError: if the configuration file does not satisfy the Pydantic constraints.
    """
    conf_loaded = get_configurations(conf_file_path=config)

    try:
        _ = check_and_build_server_config(conf_loaded=conf_loaded)
        console.log("This is a valid conf!")
    except ValidationError as e:
        console.log("This is not a valid config!")
        raise e


@app.command(name="launch")
def launch_config(
    config: Annotated[Path, typer.Argument()],
):
    """Launch the pre-processing pipeline.

    This is a Typer command to launch the pre-processing pipeline using the provided configuration file.
    Apart from the config parameter, other parameters are optional and, if given, override the associated parameter given by the parameter config.

    Parameters
    ----------
        config (Path): the Path to the configuration file.
    """
    conf_loaded = get_configurations(conf_file_path=config)

    _, root_dir, conf_dirs = check_and_build_server_config(conf_loaded=conf_loaded)

    output_dir = conf_dirs["output_dir"]
    makedirs(output_dir, exist_ok=True)

    classes = listdir(conf_dirs["input_dir"])
    stats = {}
    for cls in classes:
        # if "Normal" not in cls:
        #     continue
        input_dir = join(conf_dirs["input_dir"], cls)
        images = get_images(input_dir, ["jpg", "jpeg", "png"])
        # stats = get_images_stats(conf_dirs["input_dir"], images)
        stats.update(get_images_stats(input_dir, images))
    save_stats(stats, output_dir, "stats.csv")

    # hists = {}
    # for cls in classes:
    #     input_dir = join(conf_dirs["input_dir"], cls)
    #     images = get_images(input_dir, ["jpg", "jpeg", "png"])

    #     h = get_histograms(input_dir, images)
    #     hists[cls] = h.mean(axis=0).flatten()

    # # for cls, hist in hists.items():
    # #     print(cls, hist)
    # df = pd.DataFrame(hists)
    # df.to_csv(join(output_dir, "histograms.csv"))


if __name__ == "__main__":
    app()
