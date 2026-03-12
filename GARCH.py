from arch import arch_model
returns = df['Returns'] * 100 
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
res = garch_model.fit(disp='off')
print(res.summary())
