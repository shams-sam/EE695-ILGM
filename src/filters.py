

def kalman_filter(model, learning_rate,
                  model_mu, model_var, P_mu, P_var, 
                  std_mu, std_mu_, std_var, std_var_):
    for name, param in model.named_parameters():
        P_mu[name] = P_mu[name] + std_mu
        P_var[name] = P_var[name] + std_var
        Q_mu = P_mu[name]/(P_mu[name] + std_mu_)
        Q_var = P_var[name]/(P_var[name] + std_var_)

        model_mu[name] -= learning_rate*param.grad.cpu()
        model_mu[name] = (1-Q_mu)*model_mu[name] + Q_mu*param.detach().cpu()

        model_var[name] += (learning_rate**2)*(
            (param.grad.cpu()-model_mu[name])**2)
        model_var[name] = (1-Q_var)*model_var[name] + \
            Q_var*(param.detach().cpu()**2 - model_mu[name]**2)

        P_mu[name] = (1-Q_mu)*P_mu[name]
        P_var[name] = (1-Q_mu)*P_var[name]

    return model_mu, model_var, P_mu, P_var
