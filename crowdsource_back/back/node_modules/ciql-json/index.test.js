const ciqlJson = require(".")

describe('Tester toute les fonctions du module ciql-json', () => {

    test('La fonction Open', () => {
        expect(ciqlJson.open("yarn.lock"))
        expect(ciqlJson.open("packagef.jsn"))
        expect(ciqlJson
            .open("package.json")
            .set('location', 'abidjan')
            .set('adress.nom', '')
            .extract('{name, location, scripts, adress}')
            .getData())
    })

    test('Create ', () => {
        expect(ciqlJson.create())
        expect(ciqlJson.create({ nom: "edy" }).save())
        const data = ciqlJson.create({ nom: 'edy', prenoms: 'koffi', age: 15 }).extract("{age}").getData()
        expect(data).toEqual({ age: 15 })
    });

    test('extract and error catch', () => {
        expect(ciqlJson.create({}).extract(""))
        expect(ciqlJson.create({}).set())
        expect(
            ciqlJson
                .create({ school: {} })
                .set("school.name", "ESATIC")
                .set("school.location", "Treichville")
                .getData()
        ).toEqual({ school: { name: "ESATIC", location: "Treichville", } })
    });

    test('La fonction pushTo', () => {
        expect(
            ciqlJson
                .create({ school: "ESATIC", courses: [] })
                .pushTo("courses", "Data Sciences", "MERISE")
                .getData()
        ).toEqual({ school: "ESATIC", courses: ["Data Sciences", "MERISE"] })
    });
    test('La fonction popTo', () => {
        expect(
            ciqlJson
                .create({ school: "ESATIC", courses: ["Data Sciences", "MERISE"] })
                .popTo("courses")
                .getData()
        ).toEqual({ school: "ESATIC", courses: ["Data Sciences"] })
    });

    test('version', () => {
        let t = ciqlJson.create({}).set("1.2.3", { name: "edy" }).getData()
        expect(t).toEqual({ "1.2.3": { name: "edy" } })
    });

    test('getValues', () => {
        let t = ciqlJson.create({}).set("1.2.3", { name: "edy" }).getValues()
        expect(t).toEqual([{ name: "edy" }])
    });
    test('getKeys', () => {
        let t = ciqlJson.create({}).set("1.2.3", { name: "edy" }).getKeys()
        expect(t).toEqual(["1.2.3"])
    });
    test('save', () => {
        expect(
            ciqlJson
                .create({})
                .popTo("courses")
                .pushTo("courses","edy","koffi")
                .save()
        )
    });
    test('remove', () => {
        expect(
            ciqlJson
                .create({nom:"edy", age : 15})
                .remove("age")
                .getData()
        )
    });
})