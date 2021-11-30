from pandas import pandas as pd
import helper as helper

# pycaret report changes
from pycaret.internal.tabular import plot_model


def global_create_report(model, report_name, export_as,model_type):

    # create the markdown report
    create_markdown_report(model, report_name,model_type)

    if export_as == 'html':
        input_filename = report_name + '.md'
        output_filename = report_name + '.html'
        with open(input_filename, 'r') as f:
            html_text = markdown(f.read(), extensions=['fenced_code', 'codehilite'])
            file = open(output_filename, "w")
            file.write(html_text)
            file.close()


def create_markdown_report(model, report_name,model_type):
    # Initialize the markdown report
    mdFile = MdUtils(file_name=report_name)

    # Get report type
    report_type = helper.get_report_type(model)

    # Set the report title
    mdFile.new_header(level=1, title= + 'Model Report', style='setext')

    # Model Overview
    mdFile.new_header(level=2, title='Model Overview', style='setext')
    mdFile.new_paragraph(helper.get_model_definition(report_type))

    mdFile.new_paragraph()

    # Model Hyper parameters
    mdFile.new_header(level=2, title='Model Hyperparameters', style='setext')
    mdFile.new_paragraph(
        text="Hyper-parameters by definition are input parameters which are necessarily required by an algorithm to "
             "learn from data. They are tuned from the model itself.")

    # Model Hyper parameters
    paramlist = model.get_params()
    paramlist = pd.DataFrame.from_dict(paramlist, orient='index')
    mdFile.insert_code(paramlist.to_markdown(), language='python')
    mdFile.new_paragraph()

    # Model Plots
    mdFile.new_header(level=2, title="Model Plots")
    mdFile.new_paragraph()

    # generate the plots for this model
    model_list = helper.get_plot_list(report_type)
    for plot_type in model_list:
        plot_model(model, plot=plot_type, save=True)
        # Get Plot Name
        plot_name = helper.get_plot_name(plot_type)

        # Display header for this plot
        mdFile.new_header(level=3, title=plot_name)

        # Get Plot definition
        plot_definition = helper.get_plot_definition(plot_type)
        mdFile.new_paragraph(plot_definition)

        # Display Plot Image
        image_name = helper.get_image_name(plot_type)
        mdFile.new_line(mdFile.new_inline_image(text=plot_name, path='./' + image_name))

    # Create the markdown file
    mdFile.create_md_file()