const { login, eval_setGenesis,showFlTask,eval_startEval,eval_setFlag} = require('../controller/controller');

const express = require('express');

const router = express.Router();

router.route('/login').post(login);
router.route('/eval_setGenesis').post(eval_setGenesis);
router.route('/showFlTask').post(showFlTask);
router.route('/eval_startEval').post(eval_startEval);
router.route('/eval_setFlag').post(eval_setFlag);




module.exports.router = router;
