from server import mcp
from starlette.routing import Route, Mount

app = mcp.http_app(path="/", transport="sse")


def print_routes(app, indent=0):
    for route in app.routes:
        if isinstance(route, Route):
            print("  " * indent + f"Route: {route.path} (Methods: {route.methods})")
        elif isinstance(route, Mount):
            print("  " * indent + f"Mount: {route.path}")
            if hasattr(route.app, "routes"):
                print_routes(route.app, indent + 1)
            else:
                print("  " * (indent + 1) + "Non-Starlette app mounted")

print_routes(app)
