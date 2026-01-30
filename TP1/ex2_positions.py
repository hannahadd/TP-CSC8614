from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA


def pca_plot_and_save(position_embeddings, n_positions, out_html):
    # (n_positions, n_embd) -> numpy
    positions = position_embeddings[:n_positions].detach().cpu().numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(positions)

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        text=[str(i) for i in range(len(reduced))],
        color=list(range(len(reduced))),
        title=f"Encodages positionnels GPT-2 (PCA, positions 0-{n_positions})",
        labels={"x": "PCA 1", "y": "PCA 2"},
    )
    fig.write_html(out_html)


def main():
    model = GPT2Model.from_pretrained("gpt2")

    # learned positional embeddings
    position_embeddings = model.wpe.weight

    print("Shape position embeddings:", position_embeddings.size())
    print("n_embd:", model.config.n_embd)
    print("n_positions:", model.config.n_positions)

    pca_plot_and_save(position_embeddings, 50, "TP1/positions_50.html")
    pca_plot_and_save(position_embeddings, 200, "TP1/positions_200.html")
    print("Saved: TP1/positions_50.html, TP1/positions_200.html")


if __name__ == "__main__":
    main()
