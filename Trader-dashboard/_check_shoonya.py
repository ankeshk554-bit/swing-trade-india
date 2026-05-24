"""Check shoonya package API methods."""
import shoonya

s = shoonya.Shoonya()
methods = [m for m in dir(s) if not m.startswith('_')]
print("Shoonya methods:")
for m in methods:
    print(f"  - {m}")

# Check key methods exist
key_methods = ['login', 'place_order', 'get_quotes', 'get_history', 
               'get_order_book', 'get_positions', 'logout']
print("\nKey method check:")
for km in key_methods:
    print(f"  {'✅' if hasattr(s, km) else '❌'} {km}")
