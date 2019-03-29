from io_funcs.binary_io import  BinaryIOCollection
file_name = "/home/gyzhang/merlin/egs/slt_arctic/s1/experiments/slt_arctic_full/duration_model/inter_module/binary_label_416/arctic_a0050.lab"
io_funcs = BinaryIOCollection()
lab_values, dimension = io_funcs.load_binary_file_frame(file_name, 416)
print(lab_values[:,405:416])