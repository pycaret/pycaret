from typing import Any, Callable, Dict
import pandas as pd
import pycaret.internal.report_generator.helper as helper
from mdutils import MdUtils
from markdown import markdown
import os

from pycaret.internal.tabular import MLUsecase


def global_create_report(
    model: Any,
    report_name: str,
    export_as: str,
    model_type: MLUsecase,
    plot_model: Callable,
    get_model_id: Callable,
    get_model_name: Callable,
    available_plots: Dict[str, str],
):

    # create the markdown report
    create_markdown_report(
        model,
        report_name,
        model_type,
        plot_model,
        get_model_id,
        get_model_name,
        available_plots,
    )

    if export_as == "html":
        input_filename = f"{report_name}.md"
        output_filename = f"{report_name}.html"
        with open(input_filename, "r") as f:
            html_text = markdown(f.read(), extensions=["fenced_code", "codehilite"])
            file = open(output_filename, "w")
            file.write(html_text)
            file.close()
        os.remove(input_filename)


def create_markdown_report(
    model,
    report_name,
    report_type,
    plot_model,
    get_model_id,
    get_model_name,
    available_plots,
):
    # Initialize the markdown report
    mdFile = MdUtils(file_name=report_name)

    model_id = get_model_id(model)
    model_name = get_model_name(model)
    model_definition = helper.get_model_definition(model_id, report_type)

    # Set the report title
    mdFile.new_header(level=1, title=f"{model_name} Report", style="setext")

    # Model Overview
    mdFile.new_header(level=2, title="Model Overview", style="setext")
    if model_definition:
        mdFile.new_paragraph(model_definition)

    mdFile.new_paragraph()

    # Model Hyper parameters
    mdFile.new_header(level=2, title="Model Hyperparameters", style="setext")
    mdFile.new_paragraph(
        text="Hyper-parameters by definition are input parameters which are necessarily required by an algorithm to "
        "learn from data. They are tuned from the model itself."
    )

    # Model Hyper parameters
    paramlist = model.get_params()
    paramlist = pd.DataFrame.from_dict(paramlist, orient="index")
    mdFile.insert_code(paramlist.to_markdown(), language="python")
    mdFile.new_paragraph()

    # Model Plots
    mdFile.new_header(level=2, title="Model Plots")
    mdFile.new_paragraph()

    # generate the plots for this model
    for plot_type, plot_name in available_plots.items():
        try:
            image_name = plot_model(model, plot=plot_type, save=True)
        except:
            continue
        if not image_name:
            continue

        # Display header for this plot
        mdFile.new_header(level=3, title=plot_name)

        # Get Plot definition
        plot_definition = helper.get_plot_definition(plot_type)
        mdFile.new_paragraph(plot_definition)

        # Display Plot Image
        mdFile.new_line(mdFile.new_inline_image(text=plot_name, path="./" + image_name))

    # Create the markdown file
    mdFile.create_md_file()
