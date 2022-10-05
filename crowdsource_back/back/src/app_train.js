const express = require('express');
const cors = require('cors');
const app = express();
const { router } = require('./route/router_train');
const port = 8001;

const corsOptions = {
  origin: '*',
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use('/', router);

app.listen(port, () => {
  console.log(`Crowdsource Protocol server is listening on port ${port}`);
});
