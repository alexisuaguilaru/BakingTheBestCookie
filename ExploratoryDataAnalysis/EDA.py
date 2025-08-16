import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup:
    # Auxiliar libraries
    import marimo as mo

    # Import necessary libraries 
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell
def _():
    # Defining useful variables

    PATH = './Datasets/'
    PATH_DATASET = PATH + 'choc_chip_cookie_ingredients.csv'
    return (PATH_DATASET,)


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
        index_col=0,
    )
    CookiesRecipes.drop(columns=['Unnamed: 0'],inplace=True)
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
    return Ingredient, RecipeIndex, Unit


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
    mo.md(r"# 2. Units Standardization")
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
        axis=1,
    )

    # There are two type of values in each row/ingredient (nan values and a positive integer)
    # _UnitsByIngredients.unique()
    return


if __name__ == "__main__":
    app.run()
