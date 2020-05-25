x = range(0, 120)
_x = list(x)[::10]
_xtick_labels = ["hello,{}".format(i) for i in _x]
print(_xtick_labels)
