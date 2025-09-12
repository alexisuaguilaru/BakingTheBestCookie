import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    # Auxiliar libraries
    import marimo as mo

    # Import necessary libraries 
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt

    from scipy.spatial.distance import jaccard
    from scipy.cluster.hierarchy import linkage , dendrogram

    from sklearn.pipeline import Pipeline
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # Import auxiliar functions
    import SourceExploratoryDataAnalysis as src


@app.cell
def _():
    # Defining useful variables

    PATH = './Datasets/'
    PATH_DATASET = PATH + 'choc_chip_cookie_ingredients.csv'
    RANDOM_STATE = 8013
    return PATH_DATASET, RANDOM_STATE


@app.cell
def _():
    mo.md(r"# 1. Load Dataset and First Exploration")
    return


@app.cell
def _(NumberRecipes, RecipeIndex):
    mo.md(f"""
        There are `{NumberRecipes}` different cookie recipes which are identified by 
        its `{RecipeIndex}`. Each row of the dataset corresponds to an ingredient 
        in a cookie, therefore it is necessary to group them to describe a recipe 
        according to its ingredients.

        Using this representation, it will be possible to compare between different 
        recipes and measure their similarity based on their ingredients, that is, 
        perform a more in-depth analysis.
    """)
    return


@app.cell
def _(PATH_DATASET):
    # Loading dataset

    CookiesRecipes = pd.read_csv(
        PATH_DATASET,
        index_col = 0,
    )
    CookiesRecipes.drop(
        columns = ['Unnamed: 0'],
        inplace = True,
    )
    return (CookiesRecipes,)


@app.cell
def _(CookiesRecipes):
    mo.vstack(
        [
            mo.center(mo.md("**Examples of Dataset Instances**")),
            CookiesRecipes,
        ]
    )
    return


@app.cell
def _():
    # Defining label variables

    Ingredient = 'Ingredient'
    RecipeIndex = 'Recipe_Index'
    Quantity = 'Quantity'
    Unit = 'Unit'
    Rating = 'Rating'
    return Ingredient, Quantity, RecipeIndex, Unit


@app.cell
def _(CookiesRecipes, Ingredient, RecipeIndex, Unit):
    # Counting unique useful values

    NumberRecipes = len(CookiesRecipes[RecipeIndex].unique())
    NumberUniqueIngredients = len(CookiesRecipes[Ingredient].unique())
    NumberUniqueUnits = len(CookiesRecipes[Unit].unique())
    return (NumberRecipes,)


@app.cell
def _(CookiesRecipes, RecipeIndex):
    mo.vstack(
        [
            mo.center(mo.md("**Examples of A Cookie Recipe**")),
            CookiesRecipes.query(f"`{RecipeIndex}` == 'AR_1'"),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"## 1.1. Units Standardization")
    return


@app.cell
def _():
    mo.md(
        r"""
        For checking whether it is necessary to standardize units, the different units of measurement of an ingredietn are counted. If only one unit is used for an ingredient, then it is not necessary to standardize it.
    
        Because of there is one unit of measurement in each ingredient, it is not necessary to standardize units along the ingredients.
        """
    )
    return


@app.cell
def _(CookiesRecipes, Ingredient, RecipeIndex, Unit):
    # Counting of units of measurement for ingredient

    _IngredientsUnits = CookiesRecipes.pivot_table(
        RecipeIndex,
        Ingredient,
        Unit,
        'count',
    )

    _UnitsByIngredients = _IngredientsUnits.apply(
        lambda ingredient : len(pd.Series.unique(ingredient)),
        axis = 1,
    )

    # There are two type of values in each row/ingredient (nan values and a positive integer)
    # _UnitsByIngredients.unique()
    return


@app.cell
def _():
    mo.md(r"# 2. Cookies Ingredients")
    return


@app.cell
def _():
    mo.md(r"For each recipe cookie, the quantities of its ingredients are colleted, in order to compare two recipes based on the presence of some ingredient (or common ingredients). With this criteria, a metric or distance could be defined, which will be used to perform clustering with different techniques (this description is equivalent to use [Jaccard Distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html)). Therefore, profiles or types of cookies can be found with these clustering techniques, although these profiles might be isolated between them, because of not all cookies have the same ingredients.")
    return


@app.cell
def _(CookiesRecipes, Ingredient, Quantity, RecipeIndex):
    # Getting ingredients of each cookie

    CookiesIngredients = CookiesRecipes.pivot_table(
        Quantity,
        RecipeIndex,
        Ingredient,
        aggfunc = 'sum',
        fill_value = 0.,
    )
    return (CookiesIngredients,)


@app.cell
def _(CookiesIngredients):
    mo.vstack(
        [
            mo.center(mo.md("**Examples of Ingredients by Cookie**")),
            CookiesIngredients,
        ]
    )
    return


@app.cell
def _():
    mo.md(r"# 3. Visualization of Cookies")
    return


@app.cell
def _():
    mo.md(
        r"""
        Using [SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) for dimensionality reduction, it can be seen that the cookies are clustered or concentrated in a region. This means that there is no significative contraste between recipes.
    
        Through [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) with Jaccard distance and using different numbers of clusters, it can be observed that their Silhouette scores take low values, so there is not a good dispersion between clusters or, at least, the points are clustered in a single cluster.
    
        In the dendogram with Agglomerative Clustering, there are clusters that merge along the all possible values of distance this represents a constant growing of the clusters and therefore a unique cluster.
        """
    )
    return


@app.cell
def _(CookiesIngredients, RANDOM_STATE):
    # Applying SVD for visualization of cookies ingredientes presence

    _Standard = StandardScaler()
    _DimensionReduction = TruncatedSVD(
        n_components = 5,
        random_state = RANDOM_STATE,
    )
    _DimensionalityReduction = Pipeline(
        [
            ('Standard',_Standard),
            ('DimensionReduction',_DimensionReduction),
        ]
    )

    _ReducedDataset = _DimensionalityReduction.fit_transform(
        CookiesIngredients > 0
    )

    _fig , _axes = src.CreateCanvas()
    sns.scatterplot(
        x = _ReducedDataset[:,0],
        y = _ReducedDataset[:,1],
        ax = _axes,
        s = 24,
    )
    src.SetLabels(
        _axes,
        'PC1',
        'PC2',
        'Cookie Ingredients Visualization',
    )

    _fig
    return


@app.cell
def _(CookiesIngredients):
    _results = linkage(
        CookiesIngredients,
        'complete',
        jaccard,
    )

    _fig , _axes = src.CreateCanvas()
    dendrogram(
        _results,
        25,
        'lastp',
        ax = _axes,
    )
    src.SetLabels(
        _axes,
        'Clusters',
        'Distance',
        'Dendrogram of Hierarchical Clustering',
    )

    _fig
    return


@app.cell
def _(CookiesIngredients):
    _num_clusters = range(2,10)
    _scores = []
    for _clusters in _num_clusters:
        _clustering = AgglomerativeClustering(
            n_clusters = _clusters,
            metric = 'jaccard',
            linkage = 'complete',
        )

        _clustering.fit(CookiesIngredients>0)

        _score = silhouette_score(CookiesIngredients>0,_clustering.labels_)
        _scores.append(_score)

    _fig , _axes = src.CreateCanvas()
    _axes.plot(_num_clusters,_scores,':.b')
    src.SetLabels(
        _axes,
        'Number of Clusters',
        'Silhouette Score',
        'Silhouette Score on Different Number of Clusters',
    )

    _fig
    return


if __name__ == "__main__":
    app.run()
