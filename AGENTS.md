# Agent Instructions for GUI Screenshotting

This document outlines the steps to take a screenshot of a GUI application in this environment.

## Steps

1.  **Install Dependencies:**
    Ensure all necessary packages are installed.
    ```bash
    sudo apt-get update && sudo apt-get install -y xvfb imagemagick x11-apps x11-utils
    ```

2.  **Start the Virtual X Server (Xvfb):**
    Start a virtual display server in the background.
    ```bash
    Xvfb :0 -screen 0 1024x768x16 &
    ```
    *Note: If you get an error that the server is already active, you can likely skip this step.*

3.  **Wait for the Server to Initialize:**
    Add a small delay to ensure the X server is ready to accept connections before launching your application.
    ```bash
    sleep 2
    ```

4.  **Run Your GUI Application:**
    Launch your GUI application on the virtual display. The command below uses `xclock` as an example.
    ```bash
    DISPLAY=:0 xclock &
    ```

5.  **Take the Screenshot:**
    Use `xwd` and `imagemagick` to capture the entire screen and save it as a PNG file.
    ```bash
    xwd -display :0 -root -silent | convert xwd:- png:screenshot.png
    ```

6.  **View the Screenshot:**
    Use the `read_image_file` tool to view the captured screenshot.
    ```python
    read_image_file("screenshot.png")
    ```
