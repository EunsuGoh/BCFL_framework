import { Request, Response } from "express";
const query = require("../mysql/query/query");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const { hashRound, accessTokenSecret } = require("../config");
import { EthrDID } from "ethr-did";
import { ethers } from "ethers";
import { Wallet } from "@ethersproject/wallet";
// const { issuerPub, issuerPriv, didContractAdd } = require('../config');

const didContractAdd = process.env.DIDCONTRACTADD;
const rpcUrl = process.env.RPC_URL;
var provider = new ethers.providers.JsonRpcProvider(rpcUrl);
var contractAddress = didContractAdd; //local

const passHash = async (pass: String) => {
  const salt = await bcrypt.genSalt(Number(hashRound));
  const hashedPass = await bcrypt.hash(pass, salt);
  return hashedPass;
};

const genAccessToken = (data: any) => {
  return jwt.sign(data, accessTokenSecret, {
    expiresIn: "1d",
  });
};

export const register = async (req: Request, res: Response) => {
  const userData = req.body;

  const keypair = EthrDID.createKeyPair();
  const txSigner = new Wallet(keypair.privateKey, provider);
  const holder = new EthrDID({
    txSigner,
    provider,
    ...keypair,
    rpcUrl,
    chainNameOrId: "ganache",
    registry: contractAddress,
  });

  userData.did = holder.did;

  const hashed = await passHash(userData.password);
  userData.password = hashed;

  await query.createUser(userData, async (err: any, data1: any) => {
    if (err) {
      // error handling code goes here
      console.log("ERROR : ", err);
    }
    if (data1.affectedRows === 1) {
      await query.getUser(
        "phone_num",
        userData.phone_num,
        (err: any, data2: any) => {
          res.send({
            data: { userData: data2[0], keypair: keypair },
            msg: "Your data successfully registered!",
          });
        },
      );
    } else {
      res.send({ data: null, msg: "Your data already exists!" });
    }
  });
};

export const resignation = async (req: Request, res: Response) => {
  console.log("탈퇴 진입");
  const userData = req.body;
  console.log(userData);
  await query.deleteUser(userData, async (err: any, data: any) => {
    if (err) {
      console.error(err);
    } else {
      res.send({
        data: null,
        msg: "삭제 완료!",
      });
    }
  });
};

export const login = async (req: Request, res: Response) => {
  const loginData = req.body;
  await query.getUser(
    "user_name",
    loginData.user_name,
    async (err: any, data: any) => {
      if (err) {
        // error handling code goes here
        console.log("ERROR : ", err);
      } else {
        if (data.length === 0) {
          res.send({
            data: null,
            msg: "Wrong username or no data exists!",
          });
        } else {
          const promises = await data.map(async (elem: any) => {
            const compareBoolean = await bcrypt.compare(
              loginData.password,
              elem.password,
            );
            return compareBoolean;
          });

          const compareBoolArr = await Promise.all(promises);
          const dataFiltered = data.filter((elem: any, idx: number) => {
            return compareBoolArr[idx];
          })[0];
          if (!dataFiltered) {
            res.status(401).send({
              data: null,
              msg: "Wrong password!",
            });
          } else {
            const tokenData = {
              did: dataFiltered.did,
              phone_num: dataFiltered.phone_num,
            };

            // req.session.user_name = dataFiltered.user_name;
            // req.session.user_birth = dataFiltered.user_birth;
            // req.session.did = dataFiltered.did;
            // req.session.phone_num = dataFiltered.phone_num;

            const accessToken = genAccessToken(tokenData);
            res.send({
              data: {
                accessToken: accessToken,
                userData: dataFiltered,
              },
              msg: "Login success!",
            });
          }
        }
      }
    },
  );
};

export const storePassportVC = async (req: Request, res: Response) => {
  const { passportVC, phoneNum } = req.body;
  // 이미 있으면 업데이트(만료개념)
  // 아니면 그냥 저장
  try {
    await query.isPassport(phoneNum, async (err: any, data: any) => {
      if (data.length === 0) {
        //No data
        await query.createVC(
          "CLIENT_STORAGE_PASSPORT_VC",
          phoneNum,
          passportVC,
          (err: any, data: any) => {
            console.log(data);
            if (data.affectedRows === 1) {
              res.status(200).send({
                message: "Add passport VC success",
              });
            } else {
              res.status(400).send({
                message: "Add passport VC fail",
              });
            }
          },
        );
      } else {
        //With data
        await query.updateVC(
          "CLIENT_STORAGE_PASSPORT_VC",
          phoneNum,
          passportVC,
          (err: any, data: any) => {
            console.log(data);
            if (data.affectedRows === 1) {
              res.status(200).send({
                message: "Update passport VC success",
              });
            } else {
              res.status(400).send({
                message: "Update passport VC fail",
              });
            }
          },
        );
      }
    });
  } catch (e) {
    console.log(e);
  }
};

export const storeVisaVC = async (req: Request, res: Response) => {
  const { visaVC, phoneNum } = req.body;
  try {
    await query.createVC(
      "CLIENT_STORAGE_VISA_VC",
      phoneNum,
      visaVC,
      (err: any, data: any) => {
        console.log(data);
        if (data.affectedRows === 1) {
          res.status(200).send({ message: "Add visa VC success" });
        } else {
          res.status(400).send({ message: "Add visa VC fail" });
        }
      },
    );
  } catch (e) {
    console.log(e);
  }
};

export const storeStampVC = async (req: Request, res: Response) => {
  const { stampVC, phoneNum } = req.body;
  try {
    await query.createVC(
      "CLIENT_STORAGE_STAMP_VC",
      phoneNum,
      stampVC,
      (err: any, data: any) => {
        console.log(data);
        if (data.affectedRows === 1) {
          res.status(200).send({ message: "Add stamp VC success" });
        } else {
          res.status(400).send({ message: "Add stamp VC fail" });
        }
      },
    );
  } catch (e) {
    console.log(e);
  }
};

export const getPassportVC = async (req: Request, res: Response) => {
  const { phoneNum } = req.query;
  try {
    await query.getVC(
      "CLIENT_STORAGE_PASSPORT_VC",
      phoneNum,
      (err: any, data: any) => {
        console.log(data);
        if (data.length !== 0) {
          res.status(200).send({ data: data[0] });
        } else {
          res.status(400).send({ message: "No passport data" });
        }
      },
    );
  } catch (e) {
    console.log(e);
  }
};

export const getVisaVC = async (req: Request, res: Response) => {
  const { phoneNum } = req.query;
  try {
    await query.getVC(
      "CLIENT_STORAGE_VISA_VC",
      phoneNum,
      (err: any, data: any) => {
        console.log(data);
        if (data.length !== 0) {
          res.status(200).send({ data });
        } else {
          res.status(400).send({ message: "No visa data" });
        }
      },
    );
  } catch (e) {
    console.log(e);
  }
};

export const getStampVC = async (req: Request, res: Response) => {
  const { phoneNum } = req.query;
  try {
    await query.getVC(
      "CLIENT_STORAGE_STAMP_VC",
      phoneNum,
      (err: any, data: any) => {
        console.log(data);
        if (data.length !== 0) {
          res.status(200).send({ data });
        } else {
          res.status(400).send({ message: "No stamp data" });
        }
      },
    );
  } catch (e) {
    console.log(e);
  }
};
// module.exports = { register, authClient };