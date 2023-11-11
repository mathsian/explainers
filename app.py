from dash import Dash, html, dcc, page_registry, page_container
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

navbar = dbc.NavbarSimple(
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem(page["name"], href=page["path"])
                for page in page_registry.values()
            ],
            nav=True,
            label="Interactives"
            ),
        brand="mathsian.io",
        color="primary",
        dark=True,
        className="mb-2",
        )

app.layout = dbc.Container(
        [navbar, page_container],
        fluid=True,
        )

if __name__ == "__main__":
    app.run_server(debug=True, port=8100)
