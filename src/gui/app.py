import gradio as gr
import leafmap.plotlymap as leafmap

# def split(left, right):
#     m = leafmap.Map()  # google_map="SATELLITE"
#     # m.split_map(left_layer=left, right_layer=right)
#     naip_url = 'https://basemap.nationalmap.gov:443/arcgis/services/USGSImageryOnly/MapServer/WmsServer?'
#     m.add_wms_layer(
#         url=naip_url, layers='0', name='NAIP Imagery', format='image/png', shown=True
#     )
#     m.add_tile_layer(
#         url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
#         name="Google Satellite",
#         attribution="Google",
#     )
#     return m.to_gradio()


block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

m = leafmap.Map(zoom=4)
m.add_basemap(basemap="SATELLITE")
# m.add_controls(['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'])


def wms_map():
    m.add_tile_layer(
        url="https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}",
        name="NAIP",
        attribution="USGS",
    )
    return m


def save_map():
    m.save("./.test.png")


def build_demo():
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        with gr.Column():
            world_map = gr.Plot()

            btn = gr.Button("Save Image")

        demo.load(wms_map, [], world_map)
        btn.click(save_map, [], world_map)

    return demo


def main():
    demo = build_demo()  # gr.Interface(wms_map, inputs=[], outputs=gr.Plot(), )
    demo.launch()


if __name__ == "__main__":
    main()
