// import { Resolver } from "@ethersproject/providers";
import "dotenv/config"
import { JwtCredentialPayload,
  createVerifiableCredentialJwt,
  JwtPresentationPayload,
  createVerifiablePresentationJwt,
  verifyCredential,} from "did-jwt-vc";

const {Issuer} = require("did-jwt-vc");
const ethers = require("ethers");
const {EthrDID} = require("ethr-did");
const {Wallet} = require("@ethersproject/wallet");
const Web3 = require('web3')

import { Resolver } from "did-resolver";
import { getResolver } from "ethr-did-resolver";

const rpcURL = "http://127.0.0.1:7545"; // ganache
const provider = new ethers.providers.JsonRpcProvider(rpcURL);

const didContractAdd = process.env.DID_CONTRACT_ADDRESS

const createIssuerDID = async (issuerPub:any,issuerPriv:any) => {
  const txSigner = new Wallet(issuerPriv, provider);
  try {
    const rpcUrl = process.env.RPC_URL;
    const issuer = new EthrDID({
      txSigner,
      provider,
      identifier: issuerPub,
      privateKey: issuerPriv,
      rpcUrl,
      chainNameOrId: "ganache",
      registry: didContractAdd,
    }) as typeof Issuer;

    console.log(issuer.did);
    return issuer;
  } catch (e) {
    console.log(e);
  }
};

const createHolderDID = async (holderPub:any,holderPriv:any) => {
  // return rpcURL
  const txSigner = new Wallet(holderPriv, provider);
  try {
    const rpcUrl = process.env.RPC_URL;
    const holder = new EthrDID({
      txSigner,
      provider,
      identifier: holderPub,
      privateKey: holderPriv,
      rpcUrl,
      chainNameOrId: "ganache",
      registry: didContractAdd,
    });

    console.log(holder.did);
    return holder.did;
  } catch (e) {
    console.log(e);
  }
};

const issueVC = async (did,issuerPub,issuerPriv)=>{

  const issuer = await createIssuerDID(issuerPub,issuerPriv)

  const vcPayload : JwtCredentialPayload ={
    sub: "did:ethr:ganache:"+did,
    nbf: 1562950282,
    vc :{
      "@context": ["https://www.w3.org/2018/credentials/v1"],
      type: ["VerifiableCredential"],
      credentialSubject: {
        trainer : {
          did
        }
    }
  }
}
  const vcJwt = await createVerifiableCredentialJwt(vcPayload, issuer);
  console.log("VCJWT::" + vcJwt);
  return vcJwt
}

const verifyVC = async (vc, issuerDid) =>{
  try{

    const providerConfig = {
      name: "ganache",
      rpcUrl: rpcURL,
      registry: didContractAdd,
    };
    const ethrDidResolver = getResolver(providerConfig);
    const didResolver = new Resolver(ethrDidResolver);
    const verifiedVc = await verifyCredential(vc,didResolver)

    if (verifiedVc.payload.iss !== issuerDid) {
      console.log("False")
      return "False"
    }
    else{
      console.log("True")
      return "True"
    }
  }catch(e){
    return e
  }
  
}

module.exports = {
  createIssuerDID:createIssuerDID,
  createHolderDID:createHolderDID,
  issueVC:issueVC,
  verifyVC:verifyVC
}