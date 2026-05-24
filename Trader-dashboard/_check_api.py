"""Check shoonya API method signatures vs our bridge."""
import shoonya
import inspect

s = shoonya.Shoonya

print("=== Shoonya API Signatures ===")
methods = ['login', 'place_order', 'get_quotes', 'get_history', 
           'get_order_book', 'get_positions', 'logout', 'get_trade_book']
for m_name in methods:
    if hasattr(s, m_name):
        sig = inspect.signature(getattr(s, m_name))
        print(f"  {m_name}{sig}")
    else:
        print(f"  {m_name}: NOT FOUND ❌")

print("\n=== All public methods ===")
all_methods = [m for m in dir(s) if not m.startswith('_')]
for m in all_methods:
    print(f"  - {m}")
