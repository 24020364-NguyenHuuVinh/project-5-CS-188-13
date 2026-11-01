from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        while True:
            convergence = True
            for batch in dataloader:
                # Lấy features và label từ từ điển batch theo yêu cầu của hệ thống
                x = batch['x']
                label = batch['label']
                
                # Lấy dự đoán từ model (trả về 1 hoặc -1)
                prediction = model.get_prediction(x)
                
                # Lấy nhãn chuẩn xác
                true_label = label.item()

                # Nếu dự đoán sai, cập nhật trọng số và đánh dấu chưa hội tụ
                if prediction != true_label:
                    model.w += true_label * x
                    convergence = False
            # Nếu tất cả các dự đoán đều đúng, thoát vòng lặp
            if convergence:
                break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    # Khởi tạo bộ tối ưu hóa Adam để tự động cập nhật trọng số (learning rate = 0.01 là tối ưu)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Tạo đối tượng DataLoader để chia dữ liệu thành các batch
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Vòng lặp huấn luyện, tối đa 10000 lần lặp
    for epoch in range(10000):
        for batch in dataloader:
            # Lấy features (x) và label từ dictionary theo đúng yêu cầu đề bài
            x = batch['x']
            label = batch['label']

            # 5 bước huấn luyện tiêu chuẩn
            optimizer.zero_grad()                     # Xóa gradient cũ để tránh cộng dồn
            prediction = model(x)                     # Lấy dự đoán từ model (dạng tensor)
            loss = regression_loss(prediction, label) # Tính sai số
            loss.backward()                           # Tính gradient dựa trên sai số
            optimizer.step()                          # Cập nhật trọng số dựa trên gradient và learning rate

            with no_grad():
                updated_loss = regression_loss(model(x), label) # Tính lại sai số sau khi cập nhật trọng số
        # Kiểm tra điều kiện dừng: nếu sai số nhỏ hơn 0.02, thoát vòng lặp
        if updated_loss.item() <= 0.02:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Vòng lặp huấn luyện, tối đa 1000 lần lặp
    for epoch in range(1000):
        for batch in dataloader:
            x = batch['x']
            label = batch['label']

            # Chu trình 5 bước tối ưu hóa cơ bản
            optimizer.zero_grad()
            prediction = model(x)
            loss = digitclassifier_loss(prediction, label)
            loss.backward()
            optimizer.step()

        # Đánh giá độ chính xác sau mỗi epoch trên tập Validation
        with no_grad():
            val_accuracy = dataset.get_validation_accuracy()

        # Test Accuracy >= 97% để dừng huấn luyện
        if val_accuracy >= 0.97:
            break

def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
