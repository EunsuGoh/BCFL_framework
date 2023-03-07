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



// const provider = new Web3.providers.HttpProvider("http://127.0.0.1:7545")
// const web3 = new Web3(provider);
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
        // GOVERN_FA_PASSPORT Inner join GOVERN_USER_CLIENT
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
    // console.log(didContractAdd)
    // console.log(rpcURL)
    const providerConfig = {
      name: "ganache",
      rpcUrl: rpcURL,
      registry: didContractAdd,
    };
    const ethrDidResolver = getResolver(providerConfig);
    const didResolver = new Resolver(ethrDidResolver);
    const verifiedVc = await verifyCredential(vc,didResolver)
    // return verifiedVc
    // console.log(verifiedVc.payload.iss)
    // console.log(issuerDid)
    if (verifiedVc.payload.iss !== issuerDid) {
      console.log("False")
      return "False"
    }
    else{
      // console.log("eee")
      console.log("True")
      return "True"
    }
  }catch(e){
    return e
  }
  
}
// console.log(didContractAdd)
// console.log(verifyVC("eyJhbGciOiJFUzI1NkstUiIsInR5cCI6IkpXVCJ9.eyJ2YyI6eyJAY29udGV4dCI6WyJodHRwczovL3d3dy53My5vcmcvMjAxOC9jcmVkZW50aWFscy92MSJdLCJ0eXBlIjpbIlZlcmlmaWFibGVDcmVkZW50aWFsIl0sImNyZWRlbnRpYWxTdWJqZWN0Ijp7InRyYWluZXIiOnsiZGlkIjoiZGlkOmV0aHI6Z2FuYWNoZToweDNFNWU5MTExQWU4ZUI3OEZlMUNDM2JiODkxNWQ1RDQ2MUYzRWY5QTkifX19LCJzdWIiOiJkaWQ6ZXRocjpnYW5hY2hlOmRpZDpldGhyOmdhbmFjaGU6MHgzRTVlOTExMUFlOGVCNzhGZTFDQzNiYjg5MTVkNUQ0NjFGM0VmOUE5IiwibmJmIjoxNTYyOTUwMjgyLCJpc3MiOiJkaWQ6ZXRocjpnYW5hY2hlOjB4MEQzOGU2NTNlQzI4YmRlYTVBMjI5NmZENTk0MGFhQjJEMEI4ODc1YyJ9.lVbydxlN6Mn0ngelfdsem8S96s6qmuceA-iVmv5UO1XQfPFZB2DtiEIm5iAzTYfcKx-AGAdv9jvqEfIlPeyU4gE","did:ethr:ganache:did:ethr:ganache:0x3E5e9111Ae8eB78Fe1CC3bb8915d5D461F3Ef9A9")
// )
module.exports = {
  createIssuerDID:createIssuerDID,
  createHolderDID:createHolderDID,
  issueVC:issueVC,
  verifyVC:verifyVC
}