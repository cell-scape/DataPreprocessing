module DataPreprocessing

using CSV, DataFrames, Plots
using RollingFunctions, StatsBase, ShiftedArrays
using Downloads: download

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/"
dataset = "housing.data"
datadict = "housing.names"

download("$(url)$(dataset)", dataset)
download("$(url)$(datadict)", datadict)

# The row data is split into lines and delimited by varying amounts of whitespace
rows = readlines(open(dataset))
rowdata = map(row -> join(split(row), ","), rows)
csvdata = write(open("housing.csv", "w"), join(rowdata, "\n"))

# According to the data dictionary, the column names are as follows
columns = [:CRIM, :ZN, :INDUS, :CHAS, :NOX, :RM, :AGE, :DIS, :RAD, :TAX, :PTRATIO, :B, :LSTAT, :MEDV]

# Read the data and add the column header
df = CSV.read("housing.csv", DataFrame; header=columns)

# Descriptive statistics
size(df)
describe(df)


# Identify nominal and continuous variables
nominal = [:CHAS, :RAD]
continuous = [:CRIM, :ZN, :INDUS, :NOX, :RM, :AGE, :DIS, :TAX, :PTRATIO, :B, :LSTAT]

# Examine the distributions of the nominal variables
# Make a table of the counts of the level

histogram(df[!,:RAD], bins=0:25)
histogram(df[!, :CHAS])

radcount = counts(df[!, :RAD])
chascount = counts(df[!, :CHAS])

# Make histograms for all of the nominal variables

plot(
    histogram(df[:, :CRIM]),
    histogram(df[:, :ZN]),
    histogram(df[:, :INDUS]),
    histogram(df[:, :NOX]),
    histogram(df[:, :RM]),
    histogram(df[:, :AGE]),
    histogram(df[:, :DIS]),
    histogram(df[:, :TAX]),
    histogram(df[:, :PTRATIO]),
    histogram(df[:, :B]),
    histogram(df[:, :LSTAT]),
    histogram(df[:, :MEDV])
)

# :MEDV has a strange spike at the value 50. Try removing any exactly equal to 50.

df = df[df[:, :MEDV] .!== 50.0, :]

# Calculate the correlation every variable has with :MEDV

correlation_matrix = mapreduce(permutedims, vcat, [[corkendall(df[:, column], df[:, variable]) for variable in columns] for column in columns])

# Display it as a heat map

heatmap(correlation_matrix, xlabel=columns, ylabel=columns)

# Make a dataframe
df1 = DataFrame(correlation_matrix, columns)
df1[!, :ROWLABEL] = columns

# Sort
df2 = sort(df1, :MEDV, rev=true)

# Linear regression
cols = [:CRIM, :ZN, :INDUS, :NOX, :RM, :AGE, :DIS, :TAX, :PTRATIO, :LSTAT, :MEDV]
y = df[!, :MEDV]
X_n = Dict(col => df[!, col] for col in cols)

models = Dict(
    :CRIM => lm(@formula(MEDV ~ 1 + CRIM), df),
    :ZN => lm(@formula(MEDV ~ 1 + ZN), df),
    :INDUS => lm(@formula(MEDV ~ 1 + INDUS), df),
    :NOX => lm(@formula(MEDV ~ 1 + NOX), df),
    :RM => lm(@formula(MEDV ~ 1 + RM), df),
    :AGE => lm(@formula(MEDV ~ 1 + AGE), df),
    :DIS => lm(@formula(MEDV ~ 1 + DIS), df),
    :TAX => lm(@formula(MEDV ~ 1 + TAX), df),
    :PTRATIO => lm(@formula(MEDV ~ 1 + PTRATIO), df),
    :LSTAT => lm(@formula(MEDV ~ 1 + LSTAT), df),
    :MEDV => lm(@formula(MEDV ~ 1 + MEDV), df),
)

plots = Dict(i => scatter(X_n[i], y, smooth=true, xlabel=i, legend=false)
            for (i, model) in models)
n
plot(values(plots)..., size=(2000, 1000))

# Feature Selection! Remove B, Convert ZN to categorical, Transform CRIM
# B removed



end # module
