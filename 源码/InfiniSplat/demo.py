from __future__ import annotations

import os

os.environ.setdefault("GRADIO_SSR_MODE", "false")

from src.demo.hf_runtime import InfiniSplatRuntime
from src.demo.hf_ui import APP_CSS, APP_THEME, OUTPUT_ROOT, create_demo


runtime = InfiniSplatRuntime.load()
demo = create_demo(runtime)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(OUTPUT_ROOT)],
        max_file_size="20mb",
        show_error=False,
        ssr_mode=False,
        footer_links=[],
        theme=APP_THEME,
        css=APP_CSS,
    )
