import gradio as gr
import sys
import os

# Make sure Python can find src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gait_viz import run_pose_on_video


def process(video):
    if video is None:
        return None, "⚠️ No video uploaded", None

    try:
        # Run analysis
        out_vid, out_csv, cadence = run_pose_on_video(video)

        msg = f"✅ Processing complete!\nSaved outputs:\n📹 {out_vid}\n📊 {out_csv}"
        if cadence is not None:
            msg += f"\n\nEstimated Cadence: ~{cadence:.1f} steps/min"

        return out_vid, msg, out_csv

    except Exception as e:
        return None, f"❌ Error during processing: {str(e)}", None


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 👣 AI-Gait-Visualizer  
        Upload a short walking video, and the app will analyze gait using YOLO pose detection.  
        You’ll get an annotated video, gait metrics (stride length, cadence), and a downloadable CSV.
        """
    )

    with gr.Row():
        inp = gr.Video(label="Upload Walking Video", sources=["upload"], type="filepath")
        outv = gr.Video(label="Annotated Output")
    with gr.Row():
        msg = gr.Textbox(label="Log", lines=6)
        csv = gr.File(label="Download Metrics CSV")
    gr.Button("Run Analysis").click(process, inputs=inp, outputs=[outv, msg, csv])


if __name__ == "__main__":
    demo.launch()