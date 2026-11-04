from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()

        "*** YOUR CODE HERE ***"
        # Khởi tạo trọng số w với kích thước (1, dimensions) và tất cả các giá trị bằng 1.
        self.w = Parameter(ones(1, dimensions))



    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        # Tính tích vô hướng 
        return tensordot(x, self.w, dims=([1], [1]))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self(x)

        "*** YOUR CODE HERE ***"
        # self(x) sẽ gọi phương thức forward để tính score, sau đó dùng .item() để lấy giá trị số thực từ tensor.
        score = self(x).item()
        if score >= 0:
            return 1
        else:
            return -1
        



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        # Khởi tạo các lớp tuyến tính với kích thước đầu vào và đầu ra phù hợp
        self.layer1 = Linear(1, 64)
        self.layer2 = Linear(64, 64)
        self.layer3 = Linear(64, 1)
   

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Dữ liệu đi qua lớp 1 và bị bẻ cong bởi relu
        out = self.layer1(x)
        out = relu(out)
        
        # Dữ liệu đi qua lớp 2 và tiếp tục bị bẻ cong
        out = self.layer2(out)
        out = relu(out)
        
        # Lớp 3 xuất ra kết quả cuối cùng (giữ nguyên tính tuyến tính để ra số âm/dương)
        return self.layer3(out)
    

class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        # Khởi tạo các lớp tuyến tính với kích thước đầu vào và đầu ra phù hợp
        self.layer1 = Linear(input_size, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, output_size)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        # Tầng ẩn 1 + biến đổi phi tuyến
        out = self.layer1(x)
        out = relu(out)
        
        # Tầng ẩn 2 + biến đổi phi tuyến
        out = self.layer2(out)
        out = relu(out)
        
        # Tầng đầu ra 
        return self.layer3(out)


class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        super(LanguageIDModel, self).__init__()

        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        hidden_size = 128

        self.init_hidden = Linear(self.num_chars, hidden_size)
        self.char_to_hidden = Linear(self.num_chars, hidden_size)
        self.hidden_to_hidden = Linear(hidden_size, hidden_size, bias=False)

        self.output = Linear(hidden_size, len(self.languages))

    

    def forward(self, xs):
        """
        xs: list of tensors, each tensor has shape
            (batch_size, num_chars)

        Returns:
            Tensor of shape (batch_size, num_languages)
        """

        h = relu(self.init_hidden(xs[0]))

        for x in xs[1:]:
            h = relu(self.char_to_hidden(x) + self.hidden_to_hidden(h))

        return self.output(h)


    



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    # Tính kích thước ma trận output: H_out = H - kh + 1, W_out = W - kw + 1
    H, W = input_tensor_dimensions
    kh, kw = weight_dimensions
    out_h, out_w = H - kh + 1, W - kw + 1
    Output_Tensor = zeros((out_h, out_w))

    # Trượt kernel qua từng vùng ảnh và tính tích vô hướng
    for i in range(out_h):
        for j in range(out_w):
            patch = input[i : i + kh, j : j + kw]
            Output_Tensor[i, j] = tensordot(patch, weight, dims=([0, 1], [0, 1]))
    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        "*** YOUR CODE HERE ***"
        # Ảnh 28x28 qua kernel 3x3 thành 26x26 = 676 chiều sau khi flatten
        self.fc1 = Linear(676, 64)
        self.fc2 = Linear(64, output_size)


    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        "*** YOUR CODE HERE ***"
        # Đưa qua tầng ẩn với ReLU và xuất ra 10 logits
        h = relu(self.fc1(x))
        return self.fc2(h)


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        # 1. Tạo ma trận K, Q, V từ input bằng các layer tương ứng
        K = self.k_layer(input)
        Q = self.q_layer(input)
        V = self.v_layer(input)

        # 2. Chuyển vị ma trận K bằng movedim hoán đổi chiều thứ 2 và thứ 3
        # K ban đầu có shape (B, T, C) -> K_T có shape (B, C, T)
        K_T = movedim(K, 1, 2)

        # 3. Tính ma trận điểm số tích vô hướng chia cho sqrt(layer_size)
        M = matmul(Q, K_T) / (self.layer_size ** 0.5)

        # 4. Áp dụng Causal Mask theo đúng cú pháp hướng dẫn trong docstring
        M = M.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]

        # 5. Áp dụng softmax trên chiều cuối cùng (dim=-1)
        attention_weights = softmax(M, dim=-1)

        # 6. Nhân ma trận trọng số với V để ra kết quả đầu ra
        out = matmul(attention_weights, V)

        return out
