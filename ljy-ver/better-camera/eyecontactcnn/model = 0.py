model = 0
def a():
    global model 
    model = 114514

def b():
    global model
    print(model)

a()
b()