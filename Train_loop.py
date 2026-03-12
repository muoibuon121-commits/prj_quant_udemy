model = VolatilityLSTM(input_size=2, hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X_train, y_train = prepare_data("BTC-USD")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = qlike_loss(y_train, outputs) # Sử dụng hàm QLIKE đã viết
    # Cập nhật trọng số
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
