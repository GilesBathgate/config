# Agent Instructions for GUI Screenshotting

This document outlines the steps to take a screenshot of a GUI application in this environment.

## Steps

1.  **Install Dependencies:**
    Ensure all necessary packages are installed. `xvfb-run` is part of the `xvfb` package, `xclock` is in `x11-apps`, `xwd` is in `x11-utils`, and `convert` is in `imagemagick`.
    ```bash
    sudo apt-get update && sudo apt-get install -y xvfb imagemagick x11-apps x11-utils
    ```

2.  **Run Commands and Take Screenshot:**
    Use `xvfb-run` to automatically handle the X server setup. You can run multiple commands by passing a shell command string with `sh -c`.
    The following command will run `xclock` in the background, wait for 2 seconds, and then take a screenshot.
    ```bash
    xvfb-run sh -c "xclock & sleep 2 && xwd -root -silent | convert xwd:- png:screenshot.png"
    ```

3.  **View the Screenshot:**
    Use the `read_image_file` tool to view the captured screenshot.
    ```python
    read_image_file("screenshot.png")
    ```
