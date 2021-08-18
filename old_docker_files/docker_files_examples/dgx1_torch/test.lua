require 'cutorch'
a = torch.randn(10,10)
b = a:cuda()
print(b)
s = b:sum()
print(s)
