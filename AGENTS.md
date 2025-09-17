# Agent Instructions and Environment Notes

This document contains important information for AI agents working in this repository.

## Critical Environment Issue: The "Ghost Directory" Problem

You may encounter a critical issue in this sandbox environment where most file-based commands fail, even after a reset. This appears to be caused by the shell session being placed in a directory that has been unlinked from the filesystem.

### Symptoms

-   Commands like `ls`, `python <script>`, or `gcc <file>` fail with a `"No such file or directory"` error, even when the file clearly exists.
-   The `getcwd()` system call fails with the same error. This can be verified by compiling and running a simple C program that calls `getcwd()` and `perror()`.
-   The shell's built-in `pwd` command might still show a seemingly valid path (e.g., `/app/game_of_life`).
-   Running `ls -di .` reveals an inode number (e.g., `531`) that cannot be found when searching from the root with `find / -inum <inode_number>`. This indicates the inode is "orphaned."

### **SOLUTION**

To fix this issue for your current shell session, you **must** run the following command:

```bash
cd / && cd /app
```

This command forces the shell to discard its broken state and re-establish a valid path from the root directory. After running this command, file operations should work as expected. You can verify the fix by running `ls -di .` again and confirming that the inode number has changed.
