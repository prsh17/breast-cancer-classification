import gradio as gr
import skops.io as sio

# Load the trained SVM pipeline from file
svm_pipeline = sio.load("./Model/breast_cancer_svm_pipeline.skops", trusted=True)


def predict_cancer(
    cl_thickness,
    cell_size,
    cell_shape,
    marg_adhesion,
    epith_size,
    bare_nuclei,
    bl_cromatin,
    norm_nucleoli,
    mitoses,
):
    """Predict breast cancer based on patient features.

    Args:
        cl_thickness (int): Clump Thickness
        cell_size (int): Uniformity of Cell Size
        cell_shape (int): Uniformity of Cell Shape
        marg_adhesion (int): Marginal Adhesion
        epith_size (int): Single Epithelial Cell Size
        bare_nuclei (int): Bare Nuclei
        bl_cromatin (int): Bland Chromatin
        norm_nucleoli (int): Normal Nucleoli
        mitoses (int): Mitoses

    Returns:
        str: Predicted breast cancer class (benign or malignant)
    """
    features = [
        [
            cl_thickness,
            cell_size,
            cell_shape,
            marg_adhesion,
            epith_size,
            bare_nuclei,
            bl_cromatin,
            norm_nucleoli,
            mitoses,
        ]
    ]
    predicted_class = svm_pipeline.predict(features)[0]

    return "benign" if predicted_class == 0 else "malignant"


# Define Gradio inputs
inputs = [
    gr.Slider(minimum=1, maximum=10, step=1, label="Clump Thickness"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Uniformity of Cell Size"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Uniformity of Cell Shape"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Marginal Adhesion"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Single Epithelial Cell Size"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Bare Nuclei"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Bland Chromatin"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Normal Nucleoli"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Mitoses"),
]

# Define Gradio output
output = gr.Label(num_top_classes=2, label="class")

# Launch the Gradio interface


title = "Breast Cancer Classification"
description = "Enter the details to correctly identify cell type?"


gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=output,
    title=title,
    description=description,
    theme=gr.themes.Soft(primary_hue="rose", secondary_hue="gray"),
    css="footer {visibility: hidden}",
).launch()
