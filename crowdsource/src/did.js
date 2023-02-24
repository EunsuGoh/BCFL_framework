var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
var Issuer = require("did-jwt-vc").Issuer;
var ethers = require("ethers").ethers;
var EthrDID = require("ethr-did").EthrDID;
var Wallet = require("@ethersproject/wallet").Wallet;
var rpcURL = "http://127.0.0.1:7545"; // ganache
var provider = new ethers.providers.JsonRpcProvider(rpcURL);
var didContractAdd = process.env.DID_CONTRACT_ADDRESS;
var createIssuerDID = function (issuerPub, issuerPriv) { return __awaiter(_this, void 0, void 0, function () {
    var txSigner, rpcUrl, issuer;
    return __generator(this, function (_a) {
        txSigner = new Wallet(issuerPriv, provider);
        try {
            rpcUrl = process.env.RPC_URL;
            issuer = new EthrDID({
                txSigner: txSigner,
                provider: provider,
                identifier: issuerPub,
                privateKey: issuerPriv,
                rpcUrl: rpcUrl,
                chainNameOrId: "ganache",
                registry: didContractAdd
            });
            console.log(issuer.did);
            return [2 /*return*/, issuer];
        }
        catch (e) {
            console.log(e);
        }
        return [2 /*return*/];
    });
}); };
var createHolderDID = function (holderPub, holderPriv) { return __awaiter(_this, void 0, void 0, function () {
    var txSigner, rpcUrl, holder;
    return __generator(this, function (_a) {
        txSigner = new Wallet(holderPriv, provider);
        try {
            rpcUrl = process.env.RPC_URL;
            holder = new EthrDID({
                txSigner: txSigner,
                provider: provider,
                identifier: holderPub,
                privateKey: holderPriv,
                rpcUrl: rpcUrl,
                chainNameOrId: "ganache",
                registry: didContractAdd
            });
            console.log(holder.did);
            return [2 /*return*/, holder];
        }
        catch (e) {
            console.log(e);
        }
        return [2 /*return*/];
    });
}); };
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
    createIssuerDID: createIssuerDID,
    createHolderDID: createHolderDID
};
