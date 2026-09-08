# Application migration

Executable applications now live in the installed `jaxwind` package. Use
`jaxwind check`, `run`, `resume`, `workflow`, and `case derive`; see the
[architecture and migration guide](../doc/architecture.md).

Numerical code belongs to `src/jaxwind`, experiment analysis to `tools`, and
case data to `cases`. Old application entry points and historical checkpoint
formats are unsupported. Run all Python and verification on compute nodes.
