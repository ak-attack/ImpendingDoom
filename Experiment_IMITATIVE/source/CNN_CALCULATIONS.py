from math import floor 


#Pytorch CNN Formula:
# output_height = floor((input_height + 2*padding[0] - dilation[0]*(kernel[0]-1) - 1)/stride[0] + 1)
# output_width = floor((input_width + 2*padding[1] - dilation[1]*(kernel[1]-1) - 1)/stride[1] + 1)


def getOutputImage(input_width, input_height, kernel, stride, padding, dilation):
    def formula(input_dim):
        return floor((input_dim + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
    output_width = formula(input_width)
    output_height = formula(input_height)
    print(f"({output_width},{output_height})")
    return (output_width,output_height)

#original dim = (640, 480), scale by 10
input_width = 50
input_height = 50
kernel = 2
stride = 2
padding = 0
dilation = 1

# input_width = 30
# input_height = 45
# kernel = 3
# stride = 2
# padding = 0
# dilation = 1

output_width, output_height = getOutputImage(input_width, input_height, kernel, stride, padding, dilation)

input_width = output_width
input_height = output_height
kernel = 2
stride = 2
padding = 0
dilation = 1

output_width, output_height = getOutputImage(input_width, input_height, kernel, stride, padding, dilation)

input_width = output_width
input_height = output_height
kernel = 2
stride = 2
padding = 0
dilation = 1

output_width, output_height = getOutputImage(input_width, input_height, kernel, stride, padding, dilation)

input_width = output_width
input_height = output_height
kernel = 2
stride = 2
padding = 0
dilation = 1


output_width, output_height = getOutputImage(input_width, input_height, kernel, stride, padding, dilation)

outputlayers = 20

print(f"Flattened: {outputlayers*output_width*output_height}")
print(f"Split: {outputlayers*output_width*output_height/2}")
print(f"Split: {outputlayers*output_width*output_height/4}")