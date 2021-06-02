def foo1(filenames, params):
    print(filenames)
    print(params)
    input()


from context_menu import menus

fc = menus.FastCommand(
    "Mumax", type=".mx3", command="mumax3.exe ?x", command_vars=["FILENAME"]
)
fc.compile()
