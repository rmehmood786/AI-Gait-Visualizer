import gradio as gr
from src.gait_viz import run_pose_on_video

def process(video):
    if video is None:
        return None, "No video uploaded", None
    out_vid, out_csv, cadence = run_pose_on_video(video)
    msg = f"Saved:\n• {out_vid}\n• {out_csv}"
    if cadence is not None:
        msg += f"\nEstimated cadence: ~{cadence:.1f} steps/min (rough)"
    return out_vid, msg, out_csv

with gr.Blocks() as demo:
    gr.Markdown("# AI-Gait-Visualizer\nUpload a short walking video to visualize pose and quick gait metrics.")
    with gr.Row():
        inp = gr.Video(label="Input", sources=["upload"], height=280)
        outv = gr.Video(label="Annotated Output")
    with gr.Row():
        msg = gr.Textbox(label="Log", lines=6)
        csv = gr.File(label="Metrics CSV")
    gr.Button("Run").click(process, inputs=inp, outputs=[outv, msg, csv])

if __name__ == "__main__":
    demo.launch()