mutable struct Dense
    # Fully connected layer. By default has weights, biases, and an activation function.
    weight
    bias
    act_fn
    Dense(dim_in, dim_out, act_fn)=new(kaiming(Float64, dim_out, dim_in), 
          zeros(Float64, dim_out, 1), act_fn)
end;