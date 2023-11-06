import numpy as np
import matplotlib.pyplot as plt

def calculate_loss(p_z_given_x, q_y_given_z, labels):
    num_samples = len(labels)
    loss = 0.0
    expected_log_q_y_given_z_i_log = []
    expected_log_q_y_given_z_i_no_log = []
    
    for i in range(num_samples):
        p_z_given_x_i = p_z_given_x[i]
        q_y_given_z_i = q_y_given_z[i]
        label_i = labels[i]
        
        # Calculate the expected value of log q(y_i|z)
        expected_log_q_y_given_z_i = np.log(q_y_given_z_i[label_i])
        no_expected=q_y_given_z_i[label_i]
        # Compute the loss for the current sample
        loss_i = -np.mean(expected_log_q_y_given_z_i)
        no_log_loss=-np.mean(no_expected)
        expected_log_q_y_given_z_i_log.append(loss_i)
        expected_log_q_y_given_z_i_no_log.append(no_log_loss)

        
        # Accumulate the loss
        loss += loss_i
    
    # Calculate the average loss over all samples
    loss /= num_samples
    
    return loss, expected_log_q_y_given_z_i_log,expected_log_q_y_given_z_i_no_log

# Example usage
p_z_given_x = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])  # Example values for p(z|x)
q_y_given_z = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])  # Example values for q(y|z)
labels = np.array([0, 1, 0])  # Example labels

# Calculate the loss value and get the log values
loss_value, expected_log_q_y_given_z_i_log,expected_log_q_y_given_z_i_no_log = calculate_loss(p_z_given_x, q_y_given_z, labels)

# Print the loss value
print("Loss value:", loss_value)

# Plot the expected_log_q_y_given_z_i values with and without applying np.log
x = np.arange(len(expected_log_q_y_given_z_i_log))
y_log = expected_log_q_y_given_z_i_log
y_no_log = expected_log_q_y_given_z_i_no_log


plt.plot(x, y_log, label='With np.log')
plt.plot(x, y_no_log, label='Without np.log')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Expected Log q(y_i|z)')
plt.legend()
plt.savefig('/home/tonyhuy/bottle_classification/figure.png')  # Save the figure
# plt.show()
