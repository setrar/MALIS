# The net is simply a dictionary containing parameters,
# activations post- (A) and pre- (Z) activation function.
net=Dict("Layers"=>[], "A"=>[], "Z"=>[])

dims=[[28^2, 32] [32, 32] [32, 10]]
layers=[]
for i in 1:size(dims,2)-1
    append!(layers, [Dense(dims[1,i], dims[2,i], ReLU())])
end

head=[Dense(dims[1, size(dims,2)], dims[2, size(dims,2)], Softmax())]
append!(layers, head);
net["Layers"]=layers;