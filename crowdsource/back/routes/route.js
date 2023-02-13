// import { authClient } from '../controller/authController.js';
// import express from "express";

const {
    register,
    login,
    storePassportVC,
    storeVisaVC,
    storeStampVC,
    getPassportVC,
    getVisaVC,
    getStampVC,
    resignation,
  } = require("../controller/controller.ts");
  const express = require("express");
  const clientRoute = express.Router();
  
  // register and login process
  clientRoute.route("/login").post(login);
  clientRoute.route("/register").post(register);
  clientRoute.route("/resignation").post(resignation);
  
  // process which stores each VC
  clientRoute.route("/storePassportVC").post(storePassportVC);
  clientRoute.route("/storeVisaVC").post(storeVisaVC);
  clientRoute.route("/storeStampVC").post(storeStampVC);
  
  //process which get each VC
  clientRoute.route("/getPassportVC").get(getPassportVC);
  clientRoute.route("/getVisaVC").get(getVisaVC);
  clientRoute.route("/getStampVC").get(getStampVC);
  
  module.exports.clientRoute = clientRoute;