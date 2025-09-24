# Agent Tool Documentation

This document provides a reference for all the built-in tools available to the agent.

## Standard Tools

Standard tools use standard Python calling syntax.

*   `ls(directory_path: str = "") -> list[str]`: Lists all files and directories under the given directory (defaults to the repository root). Directories in the output will have a trailing slash (e.g., 'src/').

*   `read_file(filepath: str) -> str`: Returns the content of the specified file in the repository. It will return an error if the file does not exist.

*   `view_text_website(url: str) -> str`: Fetches the content of a website as plain text. This is useful for accessing documentation or other external resources.

*   `set_plan(plan: str) -> None`: Sets or updates the plan for how to solve the issue. Use this after initial exploration to create the first plan. If you need to revise an already approved plan, you must use this tool to set the new plan and then use `message_user` to inform the user of any significant changes.

*   `plan_step_complete(message: str) -> None`: Marks the current plan step as complete, with a message explaining what actions were taken. **Important: Before calling this tool, you must have already verified that your changes were applied correctly (e.g., by using `read_file` or `ls`).** Only call this when you have successfully completed all items needed for this plan step.

*   `message_user(message: str, continue_working: bool) -> None`: Sends a message to the user. This can be used to respond to questions, provide updates, or give feedback. Set `continue_working` to `True` if you intend to perform more actions immediately after this message. Set it to `False` if you are finished with your turn and are waiting for the next step.

*   `request_user_input(message: str) -> None`: Asks the user a question or for input and waits for a response.

*   `record_user_approval_for_plan() -> None`: Records the user's approval for the plan. Use this when the user approves the plan for the first time. If an approved plan is revised, there is no need to ask for another approval.

*   `request_code_review() -> str`: Provides a review of the current changes. You must use this tool to check for issues with your work before submitting.

*   `submit(branch_name: str, commit_message: str, title: str, description: str) -> None`: Commits the current code with a title and description (which should both be git-agnostic) and requests user approval to push to their branch. **Call this only when you are confident the code changes are complete by running all relevant tests and ensuring they pass OR when the user asks you to commit, push, submit, or otherwise finalize the code.**

*   `delete_file(filepath: str) -> str`: Deletes a file. If the file does not exist, it will return an error message.

*   `rename_file(filepath: str, new_filepath: str) -> str`: Renames and/or moves files and directories. It will return an error message if `filepath` is missing, if `new_filepath` already exists, or if the target parent directory does not exist.

*   `grep(pattern: str) -> str`: Runs grep for the given pattern.

*   `reset_all() -> None`: Resets the entire codebase to its original state. Use this tool to undo all your changes and start over.

*   `restore_file(filepath: str) -> None`: Restores the given file to its original state. Use this tool to undo all your changes to a specific file.

*   `view_image(url: str) -> Image`: Loads the image from the provided URL, allowing you to view and analyze its contents.

*   `read_pr_comments() -> str`: Reads any pending pull request comments that the user has sent for you to address.

*   `reply_to_pr_comments(replies: str) -> str`: Use this tool to reply to comments. The input must be a JSON string representing a list of objects, where each object has a "comment_id" and "reply" key.

*   `read_image_file(filepath: str) -> Image`: Reads the image file at the filepath into your context. Use this if you need to see image files on the machine, like screenshots.

*   `frontend_verification_instructions() -> str`: Returns instructions on how to write a Playwright script to verify frontend web applications and generate screenshots of your changes. You **must** call this BEFORE calling `submit` if you've made frontend web changes.

*   `frontend_verification_complete(screenshot_path: str) -> None`: Marks the frontend verification as complete, with a path to the screenshot. Only call this after `frontend_verification_instructions` has been called and you have completed the instructions there.

*   `google_search(query: str) -> str`: Performs an online Google search to retrieve the most up-to-date information. The result contains top URLs with titles and snippets. Use `view_text_website` to retrieve the full content of the relevant websites.

*   `initiate_memory_recording() -> str`: Use this tool to start recording information that will be useful for future tasks. You **must** call this tool before submitting.

## Special Tools

Special tools use a custom DSL syntax instead of a standard function call. The name of the tool should be on the first line, followed by its arguments on subsequent lines.

*   `run_in_bash_session`: Runs the given bash command in the sandbox. Successive invocations of this tool use the same bash session.

    **Example:**
    ```
    run_in_bash_session
    pip install -r requirements.txt
    ```

*   `create_file_with_block`: Use this to create a new file. If the directory does not exist, it will be created.

    **Example:**
    ```
    create_file_with_block
    path/to/new_file.py
    # This is the content of the new file.
    print("Hello, World!")
    ```

*   `overwrite_file_with_block`: Use this tool to completely replace the entire content of an existing file.

    **Example:**
    ```
    overwrite_file_with_block
    path/to/existing_file.py
    # This is the new content that will overwrite the previous file content.
    print("Hello, World!")
    ```

*   `replace_with_git_merge_diff`: Use this to perform a targeted search-and-replace to modify part of an existing file. The merge conflict markers (`<<<<<<< SEARCH`, `=======`, `>>>>>>> REPLACE`) must be exact and on their own lines.

    **Example:**
    ```
    replace_with_git_merge_diff
    path/to/file.py
    <<<<<<< SEARCH
      # old code to be replaced
    =======
      # new code to be inserted
    >>>>>>> REPLACE
    ```
