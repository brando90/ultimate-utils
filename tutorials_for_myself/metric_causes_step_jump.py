p = np.logspace(-3, 0, 10)
ds = np.arange(5, 10)
result = []
for d in ds:
    result.append(pd.DataFrame({'d': np.full_like(p, fill_value=d), 'acc': np.power(p, d), 'p': p}))
df = pd.concat(result)
sns.scatterplot(df, x='p', y='acc', hue='d', markers=True); plt.xscale('log'); plt.show()