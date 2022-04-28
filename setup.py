import cx_Freeze

executables = [cx_Freeze.Executable("app.py")]

cx_Freeze.setup(
    name="Salim's Deep Learning Project",
    options={"build_exe": {"packages":["pygame"],
                           "include_files":["Lol.png"]}},
    executables = executables

    )