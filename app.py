import logging
import random
import warnings

import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image
from gradio_imageslider import ImageSlider
from PIL import Image

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
    device = "cuda"
else:
    power_device = "CPU"
    device = "cpu"

# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
).to(device)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to(device)

MAX_SEED = 1000000
MAX_PIXEL_BUDGET = 1024 * 1024


def process_input(input_image, upscale_factor, **kwargs):
    w, h = input_image.size
    w_original, h_original = w, h
    aspect_ratio = w / h

    if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
        warnings.warn(
            f"Input image is too large ({w}x{h}). Resizing to {MAX_PIXEL_BUDGET} pixels."
        )
        input_image = input_image.resize(
            (
                int(aspect_ratio * MAX_PIXEL_BUDGET // upscale_factor),
                int(MAX_PIXEL_BUDGET // aspect_ratio // upscale_factor),
            )
        )

    # resize to multiple of 8
    w, h = input_image.size
    w = w - w % 8
    h = h - h % 8

    return input_image.resize((w, h)), w_original, h_original


# @spaces.GPU
def infer(
    seed,
    randomize_seed,
    input_image,
    num_inference_steps,
    upscale_factor,
    progress=gr.Progress(track_tqdm=True),
):
    print(input_image)
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    input_image, w_original, h_original = process_input(input_image, upscale_factor)

    print(input_image.size, w_original, h_original)

    # rescale with upscale factor
    w, h = input_image.size
    control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt="",
        control_image=control_image,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
    ).images[0]

    # resize to target desired size
    image = image.resize((w_original * upscale_factor, h_original * upscale_factor))
    image.save("output.jpg")
    # convert to numpy
    return [input_image, image]


with gr.Blocks(css=css) as demo:
    # with gr.Column(elem_id="col-container"):
    gr.Markdown(
        f"""
    # ⚡ Flux.1-dev Upscaler ControlNet ⚡
    This is an interactive demo of [Flux.1-dev Upscaler ControlNet](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler taking as input a low resolution image to generate a high resolution image.
    Currently running on {power_device}.
    """
    )

    with gr.Row():
        run_button = gr.Button(value="Run")

    with gr.Row():
        with gr.Column(scale=4):
            input_im = gr.Image(label="Input Image", type="pil")
        with gr.Column(scale=1):
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=8,
                maximum=50,
                step=1,
                value=28,
            )
            upscale_factor = gr.Slider(
                label="Upscale Factor",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

    with gr.Row():
        result = ImageSlider(label="Input / Output", type="pil")

    examples = gr.Examples(
        examples=[
            "examples/image_1.jpg",
            "examples/image_1.jpg",
            "examples/image_1.jpg",
            "examples/image_1.jpg",
        ],
        inputs=input_im,
    )

    gr.Markdown("**Disclaimer:**")
    gr.Markdown(
        "This demo is only for research purpose. Jasper cannot be held responsible for the generation of NSFW (Not Safe For Work) content through the use of this demo. Users are solely responsible for any content they create, and it is their obligation to ensure that it adheres to appropriate and ethical standards. Jasper provides the tools, but the responsibility for their use lies with the individual user."
    )
    gr.on(
        [run_button.click],
        fn=infer,
        inputs=[seed, randomize_seed, input_im, num_inference_steps, upscale_factor],
        outputs=result,
        show_api=False,
        # show_progress="minimal",
    )

demo.queue().launch(share=True)