const {Issuer} = require("did-jwt-vc");
const {ethers} = require("ethers");
const {EthrDID} = require("ethr-did");
const {Wallet} = require("@ethersproject/wallet");

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
    return holder;
  } catch (e) {
    console.log(e);
  }
};

  // const vcPayload: JwtCredentialPayload = {
  //   sub: "did:ethr:ganache:0x02d9417057f1a9aa8307a887fc5bb1499666de3e10c6affd7eb9f5f6698a36f737",
  //   nbf: 1562950282,
  //   vc: {
  //     "@context": ["https://www.w3.org/2018/credentials/v1"],
  //     type: ["VerifiableCredential"],
  //     credentialSubject: {
  //       // GOVERN_FA_PASSPORT Inner join GOVERN_USER_CLIENT
  //       passport: {
  //         passport_id,
  //         client_id,
  //         did,
  //         photo_uri,
  //         creation_date,
  //         modified_date,
  //         user_name,
  //         country_code,
  //         age,
  //         sex,
  //         birth,
  //         phone_num,
  //         personal_id,
  //       },
  //       // GOVERN_FA_VISA Inner Join GOVERN_FA_VISA_SURVEY
  //       // 두개가 따로오게? 같이오게?
  //       visa: {
  //         visa_survey_id,
  //         passport_id,
  //         visa_id,
  //         creation_date,
  //         modified_date,
  //         visa_name,
  //         visa_purpose,
  //         country_code,
  //         visa_expired_date,
  //       },
  //     },
  //   },
  // };

  // const vcJwt = await createVerifiableCredentialJwt(vcPayload, issuer);
  // console.log("VCJWT::" + vcJwt);

  // const vpPayload: JwtPresentationPayload = {
  //   vp: {
  //     "@context": ["https://www.w3.org/2018/credentials/v1"],
  //     type: ["VerifiablePresentation"],
  //     verifiableCredential: [vcJwt],
  //   },

  // const vpJwt = await createVerifiablePresentationJwt(vpPayload, issuer);
  // console.log("VPJWT::" + vpJwt);
// };

// export default createIssuerDID;
// export default createHolderDID;

module.exports = {
  createIssuerDID:createIssuerDID,
  createHolderDID:createHolderDID
}