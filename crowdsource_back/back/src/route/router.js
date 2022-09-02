const { showStakeholers, flStart } = require('../controller/controller');

const express = require('express');

const router = express.Router();

router.route('/showStakeholers').post(showStakeholers);
router.route('/flStart').post(flStart);


module.exports.router = router;
