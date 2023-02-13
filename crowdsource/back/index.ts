import express from "express";
const cors = require("cors");
// const session = require('express-session');
// const FileStore = require('session-file-store')(session);
const { clientRoute } = require("./routes/clientRoute.js");

const app = express();

const port = 4001;

const corsOptions = {
  origin: "*",
  credentials: true,
  methods: ["GET", "POST", "OPTIONS", "PATCH", "DELETE"],
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// app.use(
//     session({
//         secret: 'asefwaefawerfewrg',
//         resave: false,
//         saveUninitialized: true,
//         store: new FileStore(),
//     })
// );

app.use("/clientAuth", clientRoute);

app.listen(port, () => {
  console.log(`listening on port ${port}...`);
});