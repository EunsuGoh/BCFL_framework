const { login,showFlTask,train_participate,train_checkTrainFlag,train_startTrain} = require('../controller/controller_train');

const express = require('express');

const router = express.Router();

router.route('/login').post(login);
router.route('/showFlTask').post(showFlTask);
router.route('/train_participate').post(train_participate);
router.route('/train_checkTrainFlag').post(train_checkTrainFlag);
router.route('/train_startTrain').post(train_startTrain);



module.exports.router = router;
